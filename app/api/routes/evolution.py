from fastapi import APIRouter, Depends

from app.core.security import verifier_token_service
from app.schemas.rapports import ReponseEvolution, RequeteEvolution
from app.services.evolution import analyser_evolution

router = APIRouter(prefix="/ia", tags=["evolution"], dependencies=[Depends(verifier_token_service)])


@router.post("/evolution-sante", response_model=ReponseEvolution)
async def evolution_sante(requete: RequeteEvolution):
    return await analyser_evolution(requete)
