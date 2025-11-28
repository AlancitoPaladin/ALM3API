
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ModelResponse(BaseModel):
    id: str
    name: str
    description: str
    imageUrl: str
    modelUrl: Optional[str] = None
    modelKey: Optional[str] = None
    modelFilename: Optional[str] = None
    rating: Optional[float] = None
    price: float
    category: Optional[str] = None
    isActive: bool = True
    userId: Optional[str] = None
    detectionData: Optional[dict] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    class Config:
        from_attributes = True