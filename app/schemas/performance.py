from pydantic import BaseModel, Field


class RequetePerformanceAvs(BaseModel):
    avs_id: str | None = None  # absent = classement de tous les AVS
    jours: int = Field(default=30, ge=7, le=180)


class ScoreAvs(BaseModel):
    avs_id: str
    nom_complet: str | None = None
    score_global: float  # 0 à 100 — calcul par règles pondérées, voir services/performance_avs.py
    taux_ponctualite_rapports: float
    taux_presence_a_temps: float
    note_moyenne_appreciations: float | None = None
    nombre_rapports: int
    nombre_presences: int
    nombre_appreciations: int
    score_modele: float | None = None  # note prédite (1-5) par le modèle entraîné, si disponible — voir app/training/


class ReponsePerformanceAvs(BaseModel):
    periode_jours: int
    resultats: list[ScoreAvs]


class RequeteRechercheSemantique(BaseModel):
    requete: str = Field(..., min_length=2)
    patient_id: str | None = None  # restreint la recherche à un patient
    top_k: int = Field(default=5, ge=1, le=20)


class ResultatRecherche(BaseModel):
    type: str  # "rapport_journalier" | "message" | "patient"
    id: str
    extrait: str
    score: float
    date: str | None = None


class ReponseRechercheSemantique(BaseModel):
    resultats: list[ResultatRecherche]


class RequeteAlertesIntelligentes(BaseModel):
    patient_id: str


class AnomalieDetectee(BaseModel):
    champ: str
    valeur: str
    seuil_reference: str
    gravite: str  # "info" | "attention" | "urgent"
    explication: str
    source: str = "regle"  # "regle" (seuils fixes) | "modele" (classifieur entraîné, voir app/training/)


class ReponseAlertesIntelligentes(BaseModel):
    patient_id: str
    anomalies: list[AnomalieDetectee]
    proposition_alerte: bool
    type_alerte_propose: str | None = None
