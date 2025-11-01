from fastapi import FastAPI
from routers import rag_chat
app = FastAPI()
app.include_router(rag_chat.router)
@app.get("/")
async def index():
    return{"message":"Hello Word"}