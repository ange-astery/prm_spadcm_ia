"""
Première version de la recherche sémantique : TF-IDF + similarité
cosinus (scikit-learn), calculés à la volée sur le texte des rapports
d'un patient. Pas de vraie base vectorielle pour l'instant (voir
README.md, "faire évoluer ce modèle" — la structure est prévue pour
qu'on puisse remplacer TfidfVectorizer par de vrais embeddings de
phrases sans changer la forme de l'endpoint /ia/recherche-semantique).

Suffisant pour un volume par patient (dizaines/centaines de rapports) ;
pas conçu pour indexer TOUTE la base à chaque requête.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.performance import (
    ReponseRechercheSemantique,
    RequeteRechercheSemantique,
    ResultatRecherche,
)
from app.services import donnees


def texte_rapport(rapport: dict) -> str:
    champs = [
        rapport.get("rapportPatient"),
        rapport.get("plainte"),
        rapport.get("observations"),
        rapport.get("conclusion"),
    ]
    return " ".join(c for c in champs if c)


async def rechercher(requete: RequeteRechercheSemantique) -> ReponseRechercheSemantique:
    if not requete.patient_id:
        # V1 : recherche restreinte à un patient (voir docstring). Étendre à
        # toute la base nécessite une vraie base vectorielle indexée en
        # continu, pas un TF-IDF recalculé à la volée à chaque requête.
        return ReponseRechercheSemantique(resultats=[])

    rapports = await donnees.recuperer_rapports_patient(requete.patient_id, limite=500)
    documents = [(r, texte_rapport(r)) for r in rapports]
    documents = [(r, texte) for r, texte in documents if texte.strip()]

    if not documents:
        return ReponseRechercheSemantique(resultats=[])

    corpus = [texte for _, texte in documents] + [requete.requete]
    vectoriseur = TfidfVectorizer(max_features=2000)
    matrice = vectoriseur.fit_transform(corpus)

    vecteur_requete = matrice[-1]
    vecteurs_documents = matrice[:-1]
    similarites = cosine_similarity(vecteur_requete, vecteurs_documents)[0]

    paires = sorted(zip(documents, similarites), key=lambda p: p[1], reverse=True)
    top = paires[: requete.top_k]

    resultats = [
        ResultatRecherche(
            type="rapport_journalier",
            id=str(rapport["_id"]),
            extrait=(texte[:280] + "…") if len(texte) > 280 else texte,
            score=round(float(score), 4),
            date=rapport.get("date").date().isoformat() if hasattr(rapport.get("date"), "date") else None,
        )
        for (rapport, texte), score in top
        if score > 0
    ]

    return ReponseRechercheSemantique(resultats=resultats)
