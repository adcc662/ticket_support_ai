from fastapi import APIRouter, HTTPException
from src.schemas import (TicketResponse, TicketInput,
                     OpenAIConfig, ClassificationMethod, TrainingData,
                     CATEGORIES, PRIORITIES)
from service import classifier
from typing import List
from datetime import datetime
import uuid

router = APIRouter()

@router.post("/classify-ticket" , response_model=TicketResponse)
async def classify_ticket(ticket: TicketInput):
    try:
        result = classifier.classify_ticket(
            ticket.title,
            ticket.description,
            ticket.classification_method
        )
        ticket_id = str(uuid.uuid4())[:8]

        return TicketResponse(
            id_ticket=ticket_id,
            category=result["category"],
            priority=result["priority"],
            category_confidence=result["category_confidence"],
            priority_confidence=result["priority_confidence"],
            method_used=result["method_used"],
            detected_feeling=result.get("sentiment_detected"),
            urgency_level=result.get("urgency_level"),
            extracted_entities=result.get("extracted_entities"),
            reasoning=result.get("reasoning"),
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@router.post("/configure-openai")
async def configure_openai(config: OpenAIConfig):
    """Configure OpenAI credentials and parameters"""
    try:
        classifier.update_openai_config(config)
        return {"message": "OpenAI configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring OpenAI: {str(e)}")

@router.get("/openai-status")
async def openai_status():
    """Check OpenAI configuration status"""
    return {
        "openai_configured": classifier.openai_client is not None,
        "current_model": classifier.openai_config.model if classifier.openai_config else None,
        "available_methods": [method.value for method in ClassificationMethod]
    }

@router.post("/train-model")
async def train_model(data: TrainingData):
    """Train the model with new data"""
    try:
        success = classifier.train_models(data.tickets)
        if success:
            return {"message": "Model trained successfully", "tickets_processed": len(data.tickets)}
        else:
            raise HTTPException(status_code=500, detail="Training error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.get("/statistics")
async def get_statistics():
    """Get system statistics"""
    return {
        "available_categories": len(CATEGORIES),
        "available_priorities": len(PRIORITIES),
        "category_model_trained": classifier.category_model is not None,
        "priority_model_trained": classifier.priority_model is not None,
        "openai_configured": classifier.openai_client is not None,
        "classification_methods": [method.value for method in ClassificationMethod],
        "categories": CATEGORIES,
        "priorities": PRIORITIES
    }


@router.post("/classify-batch")
async def classify_batch(tickets: List[TicketInput]):
    """Classify multiple tickets at once"""
    results = []

    for ticket in tickets:
        try:
            result = classifier.classify_ticket(
                ticket.title,
                ticket.description,
                ticket.classification_method
            )
            import uuid
            ticket_id = str(uuid.uuid4())[:8]

            results.append(TicketResponse(
                id_ticket=ticket_id,
                category=result["category"],
                priority=result["priority"],
                category_confidence=result["category_confidence"],
                priority_confidence=result["priority_confidence"],
                method_used=result["method_used"],
                detected_feeling=result.get("sentiment_detected"),
                urgency_level=result.get("urgency_level"),
                extracted_entities=result.get("entities_extracted"),
                reasoning=result.get("reasoning"),
                timestamp=datetime.now()
            ))
        except Exception as e:
            # Log the error but continue with others
            print(f"Error processing ticket '{ticket.title}': {str(e)}")
            # Add error ticket to results for debugging
            results.append({
                "id_ticket": str(uuid.uuid4())[:8],
                "error": f"Failed to classify: {str(e)}",
                "title": ticket.title,
                "classification_method": ticket.classification_method
            })
            continue

    return {
        "tickets_processed": len(results),
        "results": results
    }