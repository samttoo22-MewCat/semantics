from pydantic import BaseModel

class NLPResponse(BaseModel):
  result: str
