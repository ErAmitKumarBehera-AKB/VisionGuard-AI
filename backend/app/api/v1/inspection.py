from typing import Optional
from io import BytesIO
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from ...schemas.inspection import (
    InspectionDetailResponse,
    InspectionHistoryResponse,
    InspectionPredictResponse,
    InspectionSummaryStats,
)
from ...services.inspection_service import InspectionService
from ...utils.database import get_db
from ...config import settings
from ...auth.security import require_admin

router = APIRouter(prefix="/inspection", tags=["Inspection"])
inspection_service = InspectionService()


@router.post(
    "/predict",
    response_model=InspectionPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit image for automated quality inspection",
)
async def predict_inspection(
    image: UploadFile = File(..., description="Manufacturing image file (PNG/JPEG)."),
    product_category: Optional[str] = Form(None, description="e.g. 'cable', 'screw', 'metal_nut', 'casting'"),
    source_dataset: Optional[str] = Form(None, description="e.g. 'mvtec', 'casting'"),
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
) -> InspectionPredictResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{image.content_type}'. Must be an image.",
        )

    image_bytes = await image.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(BytesIO(image_bytes)) as decoded:
            decoded.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        response = await inspection_service.process_inspection(
            image_bytes=image_bytes,
            filename=image.filename or "upload.png",
            db=db,
            product_category=product_category,
            source_dataset=source_dataset,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inspection inference failed: {str(e)}",
        )


@router.get(
    "/history",
    response_model=InspectionHistoryResponse,
    summary="Get paginated inspection history",
)
def get_inspection_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    prediction: Optional[str] = Query(None, description="Filter by prediction ('OK' or 'DEFECT')"),
    product_category: Optional[str] = Query(None, description="Filter by product category"),
    low_confidence_only: bool = Query(False, description="Filter only low-confidence inspections"),
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
) -> InspectionHistoryResponse:
    return inspection_service.get_history(
        db=db,
        page=page,
        page_size=page_size,
        prediction=prediction,
        product_category=product_category,
        low_confidence_only=low_confidence_only,
    )


@router.get(
    "/stats/summary",
    response_model=InspectionSummaryStats,
    summary="Get summary inspection metrics",
)
def get_inspection_summary(
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
) -> InspectionSummaryStats:
    return inspection_service.get_summary_stats(db=db)


@router.get(
    "/{inspection_id}",
    response_model=InspectionDetailResponse,
    summary="Get inspection details by ID",
)
def get_inspection_detail(
    inspection_id: str,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
) -> InspectionDetailResponse:
    record = inspection_service.get_inspection_by_id(db=db, inspection_id=inspection_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID '{inspection_id}' not found.",
        )
    return record
