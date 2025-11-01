from pydantic import BaseModel
class InputMessage(BaseModel):
    message: str