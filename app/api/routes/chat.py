from fastapi import APIRouter, Depends

from app.core.security import verifier_token_service
from app.schemas.chat import ReponseChat, RequeteChat
from app.services.chat import repondre

router = APIRouter(prefix="/ia", tags=["chat"], dependencies=[Depends(verifier_token_service)])


@router.post("/chat", response_model=ReponseChat)
async def chat(requete: RequeteChat):
    return await repondre(requete)
