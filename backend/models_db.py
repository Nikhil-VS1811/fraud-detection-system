from sqlalchemy import Column, Integer, Float, DateTime
from datetime import datetime

from database import Base

class PredictionLog(Base):

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    fraud_prediction = Column(Integer)

    fraud_probability = Column(Float)

    timestamp = Column(DateTime, default=datetime.utcnow)