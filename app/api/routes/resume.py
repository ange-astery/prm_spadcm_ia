from fastapi import APIRouter, Depends

from app.core.security import verifier_token_service
from app.schemas.rapports import ReponseResume, RequeteResume
from app.services.resume import generer_resume

router = APIRouter(prefix="/ia", tags=["resume"], dependencies=[Depends(verifier_token_service)])


@router.post("/resume-rapports", response_model=ReponseResume)
async def resume_rapports(requete: RequeteResume):
    return await generer_resume(requete)
