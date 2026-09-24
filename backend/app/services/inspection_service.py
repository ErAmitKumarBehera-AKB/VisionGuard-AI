from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid
from prometheus_client import Counter, Histogram
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..models.inspection import InspectionRecord, OperatorFeedback
from ..schemas.feedback import OperatorFeedbackRequest
from ..schemas.inspection import (
    InspectionDetailResponse,
    InspectionHistoryResponse,
    InspectionPredictResponse,
    InspectionSummaryStats,
    ProbabilityBreakdown,
)
from ..utils.logger import get_backend_logger
from .bentoml_client import BentoMLClient

logger = get_backend_logger("inspection_service")

INSPECTION_TOTAL = Counter(
    "inspection_total",
    "Total visual inspections performed",
    ["prediction", "product_category"],
)
DEFECT_PREDICTIONS_TOTAL = Counter(
    "defect_predictions_total",
    "Total defect predictions generated",
    ["product_category"],
)
LOW_CONFIDENCE_TOTAL = Counter(
    "low_confidence_predictions_total",
    "Inspections falling below the confidence threshold",
)
PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total inspection execution failures",
)
PREDICTION_CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
)
PREDICTION_LATENCY_HISTOGRAM = Histogram(
    "prediction_latency_seconds",
    "Inspection inference latency in seconds",
    buckets=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)


class InspectionService:

    def __init__(self, bentoml_client: Optional[BentoMLClient] = None) -> None:
        self.bentoml_client = bentoml_client or BentoMLClient()
        self.storage_dir = settings.image_storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def process_inspection(
        self,
        image_bytes: bytes,
        filename: str,
        db: Session,
        product_category: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> InspectionPredictResponse:
        inspection_id = f"INSP-{uuid.uuid4().hex[:12].upper()}"
        category = product_category or "generic_part"

        try:
            raw_result = await self.bentoml_client.predict(
                image_bytes=image_bytes,
                filename=filename,
                product_category=category,
                source_dataset=source_dataset,
            )

            prediction = raw_result.get("prediction", "UNKNOWN")
            confidence = float(raw_result.get("confidence", 0.0))
            latency_ms = float(raw_result.get("latency_ms", 0.0))
            model_ver = raw_result.get("model_version", "v1.0.0")
            probs = raw_result.get("probabilities", {})

            image_ext = Path(filename).suffix.lower()
            if image_ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                image_ext = ".jpg"
            image_file = self.storage_dir / f"{inspection_id}{image_ext}"
            image_file.write_bytes(image_bytes)
            image_ref = str(image_file)

            is_low_conf = confidence < settings.LOW_CONFIDENCE_THRESHOLD

            INSPECTION_TOTAL.labels(prediction=prediction, product_category=category).inc()
            if prediction == "DEFECT":
                DEFECT_PREDICTIONS_TOTAL.labels(product_category=category).inc()
            if is_low_conf:
                LOW_CONFIDENCE_TOTAL.inc()
            PREDICTION_CONFIDENCE_HISTOGRAM.observe(confidence)
            PREDICTION_LATENCY_HISTOGRAM.observe(latency_ms / 1000.0)

            db_record = InspectionRecord(
                inspection_id=inspection_id,
                timestamp=datetime.utcnow(),
                prediction=prediction,
                confidence=confidence,
                model_version=model_ver,
                latency_ms=latency_ms,
                product_category=category,
                source_dataset=source_dataset,
                image_reference=image_ref,
                feedback_status="PENDING",
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)

            prob_breakdown = None
            if probs:
                prob_breakdown = ProbabilityBreakdown(
                    OK=float(probs.get("OK", 1.0 - confidence)),
                    DEFECT=float(probs.get("DEFECT", confidence)),
                )

            return InspectionPredictResponse(
                inspection_id=inspection_id,
                prediction=prediction,
                confidence=confidence,
                model_version=model_ver,
                latency_ms=latency_ms,
                product_category=category,
                source_dataset=source_dataset,
                probabilities=prob_breakdown,
                timestamp=db_record.timestamp,
                is_low_confidence=is_low_conf,
            )

        except Exception as e:
            PREDICTION_ERRORS_TOTAL.inc()
            logger.error(f"Inspection processing failed for {inspection_id}: {e}")
            raise

    def get_history(
        self,
        db: Session,
        page: int = 1,
        page_size: int = 20,
        prediction: Optional[str] = None,
        product_category: Optional[str] = None,
        low_confidence_only: bool = False,
    ) -> InspectionHistoryResponse:
        query = db.query(InspectionRecord)

        if prediction:
            query = query.filter(InspectionRecord.prediction == prediction.upper())
        if product_category:
            query = query.filter(InspectionRecord.product_category == product_category)
        if low_confidence_only:
            query = query.filter(InspectionRecord.confidence < settings.LOW_CONFIDENCE_THRESHOLD)

        total = query.count()
        offset = (page - 1) * page_size
        records = query.order_by(InspectionRecord.timestamp.desc()).offset(offset).limit(page_size).all()

        items = [InspectionDetailResponse.model_validate(r) for r in records]

        return InspectionHistoryResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=items,
        )

    def get_inspection_by_id(self, db: Session, inspection_id: str) -> Optional[InspectionDetailResponse]:
        record = db.query(InspectionRecord).filter(InspectionRecord.inspection_id == inspection_id).first()
        if record:
            return InspectionDetailResponse.model_validate(record)
        return None

    def register_feedback(
        self,
        db: Session,
        feedback: OperatorFeedbackRequest,
    ) -> OperatorFeedback:
        record = db.query(InspectionRecord).filter(InspectionRecord.inspection_id == feedback.inspection_id).first()
        if not record:
            raise ValueError(f"Inspection record {feedback.inspection_id} not found.")

        is_corrected = record.prediction != feedback.operator_label
        record.operator_label = feedback.operator_label
        record.feedback_status = "CORRECTED" if is_corrected else "CONFIRMED"

        fb_entry = OperatorFeedback(
            inspection_id=feedback.inspection_id,
            model_prediction=record.prediction,
            operator_label=feedback.operator_label,
            confidence=record.confidence,
            model_version=record.model_version,
            comments=feedback.comments,
            created_at=datetime.utcnow(),
        )

        db.add(fb_entry)
        db.commit()
        db.refresh(fb_entry)
        db.refresh(record)

        logger.info(
            f"Feedback logged for {feedback.inspection_id}: "
            f"Model={record.prediction} | Human={feedback.operator_label} ({record.feedback_status})"
        )
        return fb_entry

    def get_summary_stats(self, db: Session) -> InspectionSummaryStats:
        total = db.query(func.count(InspectionRecord.inspection_id)).scalar() or 0
        defects = db.query(func.count(InspectionRecord.inspection_id)).filter(
            InspectionRecord.prediction == "DEFECT"
        ).scalar() or 0
        oks = db.query(func.count(InspectionRecord.inspection_id)).filter(
            InspectionRecord.prediction == "OK"
        ).scalar() or 0
        avg_lat = db.query(func.avg(InspectionRecord.latency_ms)).scalar() or 0.0
        low_conf = db.query(func.count(InspectionRecord.inspection_id)).filter(
            InspectionRecord.confidence < settings.LOW_CONFIDENCE_THRESHOLD
        ).scalar() or 0
        reviewed = db.query(func.count(InspectionRecord.inspection_id)).filter(
            InspectionRecord.feedback_status.in_(["CONFIRMED", "CORRECTED"])
        ).scalar() or 0

        defect_rate = (defects / total * 100.0) if total > 0 else 0.0

        return InspectionSummaryStats(
            total_inspections=total,
            defect_count=defects,
            ok_count=oks,
            defect_rate_percentage=round(defect_rate, 2),
            average_latency_ms=round(avg_lat, 2),
            low_confidence_count=low_conf,
            reviewed_count=reviewed,
        )
