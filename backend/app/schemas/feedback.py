from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class OperatorFeedbackRequest(BaseModel):

    inspection_id: str = Field(..., description="ID of the inspection record being reviewed.")
    operator_label: str = Field(..., pattern="^(OK|DEFECT)$", description="Human operator label: 'OK' or 'DEFECT'.")
    comments: Optional[str] = Field(None, description="Optional notes detailing defect types or visual observations.")


class OperatorFeedbackResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    feedback_id: int
    inspection_id: str
    model_prediction: str
    operator_label: str
    confidence: float
    model_version: str
    comments: Optional[str] = None
    created_at: datetime
    status: str = "RECORDED"
