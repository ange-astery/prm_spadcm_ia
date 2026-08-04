"""
Contrôle d'accès par rôle pour l'assistant IA.

IMPORTANT : ce module ne fait pas confiance au prompt/LLM pour "se
retenir" de parler d'un patient ou d'un AVS hors périmètre — il vérifie
en base, avant d'injecter quoi que ce soit dans le contexte envoyé au
LLM. Un LLM à qui on donne une donnée dans son prompt peut toujours la
répéter si on insiste assez ; la seule protection fiable est de ne
jamais lui donner une donnée que l'utilisateur n'a pas le droit de voir.

Périmètre par rôle :
- administrateur : accès à tout (patients, AVS, présences, alertes).
- coordonnateur   : uniquement les AVS et patients qu'il a lui-même
                    affectés (Assignation.assigneParId), voir donnees.py.
- medecin         : patients qu'il suit — non affiné pour l'instant
                    (pas de champ dédié côté Utilisateur/Patient), traité
                    comme "tout patient" en attendant qu'un lien
                    medecin<->patient existe côté modèle. À resserrer dès
                    que ce lien sera ajouté.
- avs             : uniquement lui-même (son propre statut/présence) et
                    les patients qui lui sont assignés.
- patient         : uniquement son propre dossier (déjà géré par
                    `chat.py` via `utilisateur.id`, en amont de ce module).
"""

from app.services import donnees


async def avs_visible_par(role: str, utilisateur_id: str, avs_cible_id: str) -> bool:
    if role == "administrateur":
        return True
    if role == "avs":
        return avs_cible_id == utilisateur_id
    if role == "coordonnateur":
        return avs_cible_id in await donnees.avs_ids_du_coordonnateur(utilisateur_id)
    return False


async def patient_visible_par(role: str, utilisateur_id: str, patient_cible_id: str) -> bool:
    if role in ("administrateur", "medecin"):
        return True
    if role == "patient":
        return patient_cible_id == utilisateur_id
    if role == "coordonnateur":
        return patient_cible_id in await donnees.patient_ids_du_coordonnateur(utilisateur_id)
    if role == "avs":
        patients_assignes = await donnees.assignations_actives_patient(patient_cible_id)
        return any(str(a["avsId"]) == utilisateur_id for a in patients_assignes)
    return False


async def filtrer_avs_autorises(role: str, utilisateur_id: str, avs_trouves: list[dict]) -> list[dict]:
    """Filtre une liste de résultats de recherche par nom : ne garde que
    ceux que ce rôle/utilisateur a le droit de consulter."""
    autorises = []
    for avs in avs_trouves:
        if await avs_visible_par(role, utilisateur_id, str(avs["_id"])):
            autorises.append(avs)
    return autorises


async def filtrer_patients_autorises(role: str, utilisateur_id: str, patients_trouves: list[dict]) -> list[dict]:
    autorises = []
    for patient in patients_trouves:
        if await patient_visible_par(role, utilisateur_id, str(patient["_id"])):
            autorises.append(patient)
    return autorises


# Rôles autorisés à demander des données agrégées / graphiques transverses
# (tous les AVS, toute une équipe) plutôt qu'une seule personne à la fois.
ROLES_VUE_GLOBALE = {"administrateur", "coordonnateur"}
