from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ProbabilityBreakdown(BaseModel):
    OK: float = Field(..., description="Probability that the sample is non-defective.")
    DEFECT: float = Field(..., description="Probability that the sample is defective.")


class InspectionPredictResponse(BaseModel):

    inspection_id: str
    prediction: str = Field(..., description="'OK' or 'DEFECT'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    latency_ms: float
    product_category: Optional[str] = None
    source_dataset: Optional[str] = None
    probabilities: Optional[ProbabilityBreakdown] = None
    timestamp: datetime
    is_low_confidence: bool = False


class InspectionDetailResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    inspection_id: str
    timestamp: datetime
    prediction: str
    confidence: float
    model_version: str
    latency_ms: float
    product_category: Optional[str] = None
    source_dataset: Optional[str] = None
    image_reference: Optional[str] = None
    operator_label: Optional[str] = None
    feedback_status: str


class InspectionHistoryResponse(BaseModel):

    total: int
    page: int
    page_size: int
    items: list[InspectionDetailResponse]


class InspectionSummaryStats(BaseModel):

    total_inspections: int
    defect_count: int
    ok_count: int
    defect_rate_percentage: float
    average_latency_ms: float
    low_confidence_count: int
    reviewed_count: int
