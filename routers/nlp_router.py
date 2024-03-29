from fastapi import APIRouter
from models.request_models import NLPRequest
from models.response_models import NLPResponse
from services.nlp_service import NLPService
from hanlp_common import document

router = APIRouter()
nlp_service = NLPService()

# Named Entity Recognition
@router.post("/ner", response_model=NLPResponse)
async def perform_nlp_task(request: NLPRequest):
  return await nlp_service.parse_ner(request.text)
  # return NLPResponse(result=result)

@router.post("/tokenise", response_model=NLPResponse)
async def perform_tokenise(request: NLPRequest):
  return await nlp_service.tokenise(request.text)

@router.post("/amr-en", response_model=NLPResponse)
async def perform_tokenise(request: NLPRequest):
  return await nlp_service.parse_amr_en(request.text)
