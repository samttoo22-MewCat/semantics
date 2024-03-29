from fastapi import FastAPI

from routers import nlp_router
from nlp import hanlp_model

app = FastAPI()

app.include_router(nlp_router.router)
