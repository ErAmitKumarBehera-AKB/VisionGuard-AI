from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ..utils.database import Base


class InspectionRecord(Base):

    __tablename__ = "inspections"

    inspection_id = Column(String(64), primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    prediction = Column(String(16), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(32), default="v1.0.0")
    latency_ms = Column(Float, nullable=False)
    product_category = Column(String(64), nullable=True, index=True)
    source_dataset = Column(String(32), nullable=True)
    image_reference = Column(String(256), nullable=True)
    operator_label = Column(String(16), nullable=True)
    feedback_status = Column(String(32), default="PENDING", index=True)

    feedbacks = relationship("OperatorFeedback", back_populates="inspection", cascade="all, delete-orphan")


class OperatorFeedback(Base):

    __tablename__ = "feedbacks"

    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(String(64), ForeignKey("inspections.inspection_id"), nullable=False, index=True)
    model_prediction = Column(String(16), nullable=False)
    operator_label = Column(String(16), nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("InspectionRecord", back_populates="feedbacks")
