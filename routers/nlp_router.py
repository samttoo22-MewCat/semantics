from fastapi import APIRouter
from models.request_models import NLPRequest
from models.response_models import NLPResponse
from services.nlp_service import NLPService
from hanlp_common import document

router = APIRouter()
nlp_service = NLPService()

@router.post("/ner", response_model=NLPResponse)
async def perform_nlp_task(request: NLPRequest):
  result = await nlp_service.parse_ner(request.text)
  return NLPResponse(result=result)

@router.post("/tokenise", response_model=NLPResponse)
async def perform_tokenise(request: NLPRequest):
  result = await nlp_service.tokenise(request.text)
  return NLPResponse(result=result)

@router.post("/amr-en", response_model=NLPResponse)
async def perform_tokenise(request: NLPRequest):
  result = await nlp_service.parse_amr_en(request.text)
  return NLPResponse(result=result)

@router.post("/con", response_model=NLPResponse)
async def perform_con(request: NLPRequest):
  result = await nlp_service.con(request.text)
  return NLPResponse(result=result)

@router.post("/dep", response_model=NLPResponse)
async def perform_dep(request: NLPRequest):
  result = await nlp_service.dep(request.text)
  return NLPResponse(result=result)

@router.post("/sdp", response_model=NLPResponse)
async def perform_sdp(request: NLPRequest):
  result = await nlp_service.sdp(request.text)
  return NLPResponse(result=result)

@router.post("/ud", response_model=NLPResponse)
async def perform_ud(request: NLPRequest):
  result = await nlp_service.ud(request.text)
  return NLPResponse(result=result)
