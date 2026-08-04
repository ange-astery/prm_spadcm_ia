from fastapi import APIRouter, Depends

from app.core.security import verifier_token_service
from app.schemas.performance import ReponseAlertesIntelligentes, RequeteAlertesIntelligentes
from app.services.alertes import detecter_anomalies

router = APIRouter(prefix="/ia", tags=["alertes"], dependencies=[Depends(verifier_token_service)])


@router.post("/alertes-intelligentes", response_model=ReponseAlertesIntelligentes)
async def alertes_intelligentes(requete: RequeteAlertesIntelligentes):
    return await detecter_anomalies(requete)
