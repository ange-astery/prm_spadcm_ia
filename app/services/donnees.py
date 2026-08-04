"""
Accès en lecture seule aux collections MongoDB partagées avec
prm_spadcm_backend. Les noms de collection suivent exactement ceux du
backend Node (voir models/*.js) : certains sont explicites
(`collection: '...'` dans le schéma Mongoose), d'autres suivent le
pluriel par défaut de Mongoose (nom du modèle en minuscules + "s").
"""

from datetime import date, datetime, timedelta

from bson import ObjectId

from app.db.mongo import get_database

COL_UTILISATEURS = "utilisateurs"
COL_PATIENTS = "patients"
COL_RAPPORTS = "rapportjournaliers"  # RapportJournalier -> pas de collection explicite côté Node
COL_PRESENCES = "presences"
COL_APPRECIATIONS = "appreciations"
COL_ALERTES = "alertes"
COL_ASSIGNATIONS = "assignations"  # Assignation -> pluriel par défaut de Mongoose


def _vers_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Identifiant Mongo invalide: {id_str}") from exc


async def recuperer_patient(patient_id: str) -> dict | None:
    db = get_database()
    return await db[COL_PATIENTS].find_one({"_id": _vers_object_id(patient_id)})


async def recuperer_rapports_patient(
    patient_id: str,
    date_debut: date | None = None,
    date_fin: date | None = None,
    limite: int = 200,
) -> list[dict]:
    db = get_database()
    filtre: dict = {"patientId": _vers_object_id(patient_id)}

    bornes_date: dict = {}
    if date_debut:
        bornes_date["$gte"] = datetime.combine(date_debut, datetime.min.time())
    if date_fin:
        bornes_date["$lte"] = datetime.combine(date_fin, datetime.max.time())
    if bornes_date:
        filtre["date"] = bornes_date

    curseur = db[COL_RAPPORTS].find(filtre).sort("date", -1).limit(limite)
    return [doc async for doc in curseur]


async def recuperer_rapports_recents(patient_id: str, jours: int = 7) -> list[dict]:
    date_debut = date.today() - timedelta(days=jours)
    return await recuperer_rapports_patient(patient_id, date_debut=date_debut)


async def recuperer_avs(avs_id: str) -> dict | None:
    db = get_database()
    return await db[COL_UTILISATEURS].find_one({"_id": _vers_object_id(avs_id), "role": "avs"})


async def lister_avs() -> list[dict]:
    db = get_database()
    curseur = db[COL_UTILISATEURS].find({"role": "avs"})
    return [doc async for doc in curseur]


async def recuperer_rapports_avs(avs_id: str, jours: int) -> list[dict]:
    db = get_database()
    date_debut = datetime.combine(date.today() - timedelta(days=jours), datetime.min.time())
    curseur = db[COL_RAPPORTS].find({"avsId": _vers_object_id(avs_id), "date": {"$gte": date_debut}})
    return [doc async for doc in curseur]


async def recuperer_presences_avs(avs_id: str, jours: int) -> list[dict]:
    db = get_database()
    date_debut = datetime.combine(date.today() - timedelta(days=jours), datetime.min.time())
    curseur = db[COL_PRESENCES].find({"avsId": _vers_object_id(avs_id), "date": {"$gte": date_debut}})
    return [doc async for doc in curseur]


async def recuperer_appreciations_avs(avs_id: str, jours: int) -> list[dict]:
    db = get_database()
    date_debut = datetime.combine(date.today() - timedelta(days=jours), datetime.min.time())
    curseur = db[COL_APPRECIATIONS].find({"avsId": _vers_object_id(avs_id), "createdAt": {"$gte": date_debut}})
    return [doc async for doc in curseur]


# --------------------------------------------------------------------------- #
# Recherche par nom — indispensable pour un chat en langage naturel : un
# coordonnateur/admin tape "Avs SPAD" ou "Mme Nguena", pas un ObjectId Mongo.
# Insensible à la casse, tolérant sur prénom/nom (voir intention.py, qui
# extrait ce texte libre depuis le message avant d'appeler ces fonctions).
# --------------------------------------------------------------------------- #

async def rechercher_avs_par_nom(nom: str, limite: int = 5) -> list[dict]:
    db = get_database()
    regex = {"$regex": nom.strip(), "$options": "i"}
    curseur = db[COL_UTILISATEURS].find(
        {"role": "avs", "$or": [{"prenom": regex}, {"nom": regex}]}
    ).limit(limite)
    return [doc async for doc in curseur]


async def rechercher_patient_par_nom(nom: str, limite: int = 5) -> list[dict]:
    db = get_database()
    regex = {"$regex": nom.strip(), "$options": "i"}
    curseur = db[COL_PATIENTS].find(
        {"$or": [{"prenom": regex}, {"nom": regex}]}
    ).limit(limite)
    return [doc async for doc in curseur]


# --------------------------------------------------------------------------- #
# Statut "temps réel" d'un AVS ou d'un patient — ce que demande typiquement
# un coordonnateur ("est-ce que l'AVS X est à l'heure ?", "où en est
# l'intervention chez Y ?"). Combine présence du jour (pointage) et
# affectations actives, comme le fait déjà la rubrique "Suivi des
# interventions" du tableau de bord (voir statsController.js côté Node).
# --------------------------------------------------------------------------- #

async def statut_du_jour_avs(avs_id: str) -> dict:
    db = get_database()
    aujourd_hui = datetime.combine(date.today(), datetime.min.time())
    presence = await db[COL_PRESENCES].find_one({"avsId": _vers_object_id(avs_id), "date": aujourd_hui})
    assignations = [
        doc async for doc in db[COL_ASSIGNATIONS].find({"avsId": _vers_object_id(avs_id), "statut": "active"})
    ]
    return {"presence": presence, "assignations_actives": assignations}


async def assignations_actives_patient(patient_id: str) -> list[dict]:
    db = get_database()
    curseur = db[COL_ASSIGNATIONS].find({"patientId": _vers_object_id(patient_id), "statut": "active"})
    return [doc async for doc in curseur]


# --------------------------------------------------------------------------- #
# Périmètre d'un coordonnateur — ses AVS et ses patients sont ceux qu'il a
# lui-même affectés (Assignation.assigneParId). Sert à la fois à la
# recherche par nom scopée (permissions.py) et à une liste directe
# ("quels sont mes AVS ?").
# --------------------------------------------------------------------------- #

async def avs_ids_du_coordonnateur(coordonnateur_id: str) -> set[str]:
    db = get_database()
    curseur = db[COL_ASSIGNATIONS].find(
        {"assigneParId": _vers_object_id(coordonnateur_id)}, {"avsId": 1}
    )
    return {str(doc["avsId"]) async for doc in curseur}


async def patient_ids_du_coordonnateur(coordonnateur_id: str) -> set[str]:
    db = get_database()
    curseur = db[COL_ASSIGNATIONS].find(
        {"assigneParId": _vers_object_id(coordonnateur_id)}, {"patientId": 1}
    )
    return {str(doc["patientId"]) async for doc in curseur}
