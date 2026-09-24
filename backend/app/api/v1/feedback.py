from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...models.inspection import OperatorFeedback
from ...schemas.feedback import OperatorFeedbackRequest, OperatorFeedbackResponse
from ...services.inspection_service import InspectionService
from ...utils.database import get_db
from ...auth.security import require_admin

router = APIRouter(prefix="/feedback", tags=["Human Feedback"])
inspection_service = InspectionService()


@router.post(
    "",
    response_model=OperatorFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit human operator validation or correction",
)
def submit_feedback(
    feedback: OperatorFeedbackRequest,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
) -> OperatorFeedbackResponse:
    try:
        fb_entry = inspection_service.register_feedback(db=db, feedback=feedback)
        return OperatorFeedbackResponse.model_validate(fb_entry)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


@router.get(
    "/export",
    summary="Export validated feedback records for retraining pipeline",
)
def export_feedback_dataset(
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
) -> dict:
    records = db.query(OperatorFeedback).all()
    feedback_data = [
        {
            "feedback_id": r.feedback_id,
            "inspection_id": r.inspection_id,
            "model_prediction": r.model_prediction,
            "human_label": r.operator_label,
            "confidence": r.confidence,
            "model_version": r.model_version,
            "comments": r.comments,
            "timestamp": r.created_at.isoformat(),
        }
        for r in records
    ]
    return {
        "total_records": len(feedback_data),
        "items": feedback_data,
        "ready_for_dvc_ingestion": len(feedback_data) > 0,
    }
