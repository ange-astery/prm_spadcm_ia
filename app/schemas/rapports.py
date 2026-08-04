from datetime import date

from pydantic import BaseModel, Field


class RequeteResume(BaseModel):
    patient_id: str
    # Bornes optionnelles ; par défaut, les 7 derniers jours (voir service).
    date_debut: date | None = None
    date_fin: date | None = None
    ecrire_dans_rapport_id: str | None = Field(
        default=None,
        description=(
            "Si fourni, ce service tente un callback PATCH vers le backend "
            "Node pour remplir RapportJournalier.resumeIA de ce rapport "
            "précis (voir README.md)."
        ),
    )


class ReponseResume(BaseModel):
    patient_id: str
    periode: str
    resume: str
    nombre_rapports_analyses: int


class RequeteEvolution(BaseModel):
    patient_id: str
    jours: int = Field(default=30, ge=7, le=180)


class PointEvolution(BaseModel):
    date: date
    pouls_moyen: float | None = None
    temperature_moyenne: float | None = None
    spo2_moyen: float | None = None


class ReponseEvolution(BaseModel):
    patient_id: str
    points: list[PointEvolution]
    tendance: str  # ex: "stable", "amélioration", "dégradation à surveiller"
    analyse: str  # courte synthèse en langage naturel
