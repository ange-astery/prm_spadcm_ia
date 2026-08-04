from fastapi import APIRouter, Depends

from app.core.security import verifier_token_service
from app.schemas.performance import ReponsePerformanceAvs, RequetePerformanceAvs
from app.services.performance_avs import calculer_performance

router = APIRouter(prefix="/ia", tags=["performance"], dependencies=[Depends(verifier_token_service)])


@router.post("/performance-avs", response_model=ReponsePerformanceAvs)
async def performance_avs(requete: RequetePerformanceAvs):
    return await calculer_performance(requete)
