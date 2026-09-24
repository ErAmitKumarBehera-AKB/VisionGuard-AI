from dataclasses import dataclass
from ..config import settings
@dataclass(frozen=True)
class Decision: status:str; decision:str; review_required:bool; final_label:str|None
def decide(prediction:str, confidence:float)->Decision:
    prediction=prediction.upper()
    if prediction not in {"OK", "DEFECT"}:
        raise ValueError("prediction must be OK or DEFECT")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if confidence < settings.LOW_CONFIDENCE_THRESHOLD: return Decision("PENDING_REVIEW","PENDING_HUMAN_REVIEW",True,None)
    return Decision("COMPLETED","AUTOMATIC_OK",False,"OK") if prediction=="OK" else Decision("COMPLETED","AUTOMATIC_DEFECT",False,"DEFECT")
