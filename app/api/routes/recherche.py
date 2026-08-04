from fastapi import APIRouter, Depends

from app.core.security import verifier_token_service
from app.schemas.performance import ReponseRechercheSemantique, RequeteRechercheSemantique
from app.services.recherche_semantique import rechercher

router = APIRouter(prefix="/ia", tags=["recherche"], dependencies=[Depends(verifier_token_service)])


@router.post("/recherche-semantique", response_model=ReponseRechercheSemantique)
async def recherche_semantique(requete: RequeteRechercheSemantique):
    return await rechercher(requete)
