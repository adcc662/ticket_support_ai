from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class ClassificationMethod(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"
    HYBRID = "hybrid"

class TicketInput(BaseModel):
    title: str
    description: str
    user_email: Optional[str]
    classification_method: Optional[ClassificationMethod]

class TicketResponse(BaseModel):
    id_ticket: str
    category: str
    priority: str
    category_confidence: float
    priority_confidence: float
    method_used: str
    detected_feeling: Optional[str] = None
    urgency_level: Optional[str] = None
    extracted_entities: Optional[str] = None
    reasoning: Optional[str] = None
    timestamp: datetime

class TrainingData(BaseModel):
    tickets: List[dict]

class OpenAIConfig(BaseModel):
    api_key: str
    model: Optional[str] = "gpt-5"
    temperature: Optional[float] = 1

CATEGORIES = [
    "Hardware", "Software", "Network", "Login",
    "Billing", "General", "Email", "Security"
]

PRIORITIES = ["High", "Medium", "Low"]
