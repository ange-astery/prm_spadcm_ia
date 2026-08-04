"""
Assistant IA du PRM SPAD -- orchestrateur en 2 passes.

  1. intention.py  : le LLM identifie CE QUE demande l'utilisateur
                      (statut d'un AVS/patient precis, graphique, ou
                      question generale) a partir du texte libre.
  2. permissions.py : on verifie que ce role a le droit de voir cette
                      donnee precise (jamais de confiance aveugle au
                      prompt pour ca).
  3. donnees.py / graphiques.py : on va chercher UNIQUEMENT les donnees
                      autorisees et pertinentes dans MongoDB.
  4. llm.py         : un dernier appel redige la reponse en francais a
                      partir de ces donnees reelles (jamais inventees).

Le cas "patient qui suit son propre dossier" (contexte automatique via
`utilisateur.id`) reste gere comme avant : c'est le chemin le plus
frequent et le plus simple, pas la peine de le faire passer par
l'extraction d'intention.
"""

from app.core.config import get_settings
from app.schemas.chat import ReponseChat, RequeteChat
from app.services import donnees, graphiques, permissions
from app.services.intention import extraire_intention
from app.services.llm import generer_texte
from app.services.recherche_semantique import texte_rapport  # reutilise tel quel

ROLE_LISIBLE = {
    "avs": "un(e) Auxiliaire de Vie Sociale (AVS) en tournee chez des patients",
    "medecin": "un medecin qui supervise des dossiers patients",
    "coordonnateur": "un coordonnateur qui supervise les AVS et le suivi des patients",
    "administrateur": "un administrateur de la plateforme, qui peut consulter toutes les donnees du projet",
    "patient": "un patient ou un membre de sa famille suivant son dossier de soins a domicile",
}


def _formater_presence(presence: dict | None) -> str:
    if not presence:
        return "Aucun pointage enregistre aujourd'hui pour cet AVS."
    statut = presence.get("statut", "inconnu")
    check_in = presence.get("heureCheckIn")
    check_out = presence.get("heureCheckOut")
    morceaux = [f"Statut du pointage aujourd'hui : {statut}."]
    if check_in:
        morceaux.append(f"Arrivee enregistree a {check_in.strftime('%Hh%M')}.")
    if check_out:
        morceaux.append(f"Depart enregistre a {check_out.strftime('%Hh%M')}.")
    return " ".join(morceaux)


def _formater_assignations(assignations: list[dict]) -> str:
    if not assignations:
        return "Aucune affectation active en ce moment."
    return "Affectations actives : " + "; ".join(
        f"patient {a.get('patientId')}, jours {', '.join(a.get('joursTravail') or []) or 'non precises'}"
        for a in assignations
    )


async def _contexte_patient(patient_id: str) -> tuple[str, list[str]]:
    settings = get_settings()
    patient = await donnees.recuperer_patient(patient_id)
    if not patient:
        return "", []

    rapports = (await donnees.recuperer_rapports_recents(patient_id, jours=7))[: settings.rag_top_k]

    morceaux = [
        f"Patient suivi : {patient.get('prenom', '')} {patient.get('nom', '')}, "
        f"pathologie principale : {patient.get('pathologie') or 'non renseignee'}."
    ]
    sources = [f"Dossier patient {patient_id}"]

    for rapport in rapports:
        texte = texte_rapport(rapport)
        if texte:
            date_rapport = rapport.get("date")
            date_str = date_rapport.date().isoformat() if hasattr(date_rapport, "date") else str(date_rapport)
            morceaux.append(f"Rapport du {date_str} : {texte}")
            sources.append(f"Rapport journalier du {date_str}")

    return "\n".join(morceaux), sources


async def _contexte_statut_avs(role: str, utilisateur_id: str, nom_avs: str) -> tuple[str, list[str]]:
    candidats = await donnees.rechercher_avs_par_nom(nom_avs)
    candidats = await permissions.filtrer_avs_autorises(role, utilisateur_id, candidats)

    if not candidats:
        return (
            f"Aucun AVS nomme \u00ab {nom_avs} \u00bb n'a ete trouve dans le perimetre de cet utilisateur "
            "(soit le nom est incorrect, soit cet AVS n'est pas dans son equipe).",
            [],
        )

    morceaux, sources = [], []
    for avs in candidats[:3]:
        avs_id = str(avs["_id"])
        statut = await donnees.statut_du_jour_avs(avs_id)
        nom_complet = f"{avs.get('prenom', '')} {avs.get('nom', '')}".strip()
        morceaux.append(
            f"AVS {nom_complet} : {_formater_presence(statut['presence'])} "
            f"{_formater_assignations(statut['assignations_actives'])}"
        )
        sources.append(f"Pointage + affectations du jour -- AVS {nom_complet}")

    return "\n".join(morceaux), sources


async def _contexte_statut_patient(role: str, utilisateur_id: str, nom_patient: str) -> tuple[str, list[str]]:
    candidats = await donnees.rechercher_patient_par_nom(nom_patient)
    candidats = await permissions.filtrer_patients_autorises(role, utilisateur_id, candidats)

    if not candidats:
        return (
            f"Aucun patient nomme \u00ab {nom_patient} \u00bb n'a ete trouve dans le perimetre de cet utilisateur.",
            [],
        )

    morceaux, sources = [], []
    for patient in candidats[:3]:
        contexte, src = await _contexte_patient(str(patient["_id"]))
        assignations = await donnees.assignations_actives_patient(str(patient["_id"]))
        morceaux.append(contexte + " " + _formater_assignations(assignations))
        sources.extend(src)

    return "\n".join(morceaux), sources


async def repondre(requete: RequeteChat) -> ReponseChat:
    utilisateur = requete.utilisateur
    role_lisible = ROLE_LISIBLE.get(utilisateur.role, "un utilisateur")

    contexte, sources, donnees_graphique = "", [], None

    # Chemin le plus frequent, inchange : un patient/famille qui consulte
    # son propre dossier, ou un AVS/medecin/coordonnateur qui a ouvert un
    # dossier patient precis dans l'app au moment de poser sa question.
    patient_id_direct = requete.patient_id or (
        utilisateur.id if utilisateur.role == "patient" else None
    )
    if patient_id_direct:
        contexte, sources = await _contexte_patient(patient_id_direct)

    # Sinon, on laisse le LLM identifier l'intention a partir du texte
    # libre (nom d'AVS/patient cite, demande de graphique, etc.).
    if not contexte:
        intention = await extraire_intention(requete.message)

        if intention.intention == "statut_avs" and intention.nom_avs:
            contexte, sources = await _contexte_statut_avs(utilisateur.role, utilisateur.id, intention.nom_avs)
        elif intention.intention == "statut_patient" and intention.nom_patient:
            contexte, sources = await _contexte_statut_patient(utilisateur.role, utilisateur.id, intention.nom_patient)
        elif intention.intention == "graphique":
            if intention.nom_avs:
                candidats = await permissions.filtrer_avs_autorises(
                    utilisateur.role, utilisateur.id, await donnees.rechercher_avs_par_nom(intention.nom_avs)
                )
                if candidats:
                    donnees_graphique = await graphiques.graphique_presences_avs(
                        str(candidats[0]["_id"]), intention.periode_jours
                    )
            else:
                donnees_graphique = await graphiques.graphique_performance_avs(
                    utilisateur.role, utilisateur.id, intention.periode_jours
                )
            if donnees_graphique:
                contexte = f"Un graphique a ete prepare : {donnees_graphique['titre']}."
                sources = ["Statistiques de performance/presence AVS"]

    system_prompt = (
        "Tu es l'assistant IA integre au PRM de SPAD Cameroun (soins et "
        "prestations a domicile). Tu reponds en francais, de facon breve, "
        f"concrete et bienveillante. La personne qui te parle est {role_lisible} "
        f"(prenom : {utilisateur.prenom}). Tu ne poses jamais de diagnostic et "
        "tu orientes vers le coordonnateur ou le medecin referent pour toute "
        "decision medicale precise. Si un graphique a ete prepare, dis-le "
        "brievement sans repeter tous les chiffres (l'utilisateur le voit "
        "s'afficher juste apres ton message)."
        + (
            f"\n\nDonnees reelles du projet (utilise-les si pertinentes, "
            f"ne les invente jamais si absentes) :\n{contexte}"
            if contexte
            else "\n\nAucune donnee precise du projet n'a ete trouvee pour cette question : "
            "reponds de facon generale ou invite la personne a preciser un nom."
        )
    )

    messages = [{"role": m.role, "content": m.contenu} for m in requete.historique[-20:]]
    messages.append({"role": "user", "content": requete.message})

    reponse = await generer_texte(system_prompt=system_prompt, messages=messages)

    return ReponseChat(reponse=reponse, sources=sources, donnees_graphique=donnees_graphique)
