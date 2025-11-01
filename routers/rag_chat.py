# routes/chat_routes.py
from fastapi import APIRouter
from schema.rag_schema import InputMessage
from services import rag_service

router = APIRouter()

@router.post("/rag-chat")
def ai_chat(data_in: InputMessage):
    response = rag_service.get_chat_response(data_in)
    return {"response": response}
