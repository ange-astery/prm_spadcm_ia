"""
Construit des données de graphique prêtes à consommer par fl_chart côté
Flutter (`prm_spadcm_personnel`), à partir des calculs qui existent déjà
(performance_avs.py) plutôt que de faire produire des chiffres par le
LLM — un LLM ne doit jamais être la source des nombres affichés dans un
graphique, seulement décider QUEL graphique répond à la question.

Format de sortie volontairement générique (voir ReponseChat.donnees_graphique
dans schemas/chat.py) : {"type", "titre", "labels", "series": [{"nom","valeurs"}]}
— un seul format que le widget Flutter sait interpréter, quel que soit le
type de graphique (barres, ligne).
"""

from app.schemas.performance import RequetePerformanceAvs
from app.services import donnees, permissions
from app.services.performance_avs import calculer_performance


async def graphique_performance_avs(role: str, utilisateur_id: str, jours: int) -> dict | None:
    """Classement des AVS (ponctualité rapports/présence) — réservé aux
    rôles à vue globale (voir permissions.ROLES_VUE_GLOBALE) ; un AVS qui
    demande "un graphique" reçoit son propre historique, pas le classement
    de toute l'équipe."""
    if role in permissions.ROLES_VUE_GLOBALE:
        resultat = await calculer_performance(RequetePerformanceAvs(jours=jours))
    elif role == "avs":
        resultat = await calculer_performance(RequetePerformanceAvs(avs_id=utilisateur_id, jours=jours))
    else:
        return None

    if not resultat.resultats:
        return None

    top = resultat.resultats[:10]  # lisible sur un écran mobile
    return {
        "type": "bar",
        "titre": f"Ponctualité des AVS — {jours} derniers jours",
        "labels": [r.nom_complet or r.avs_id[-6:] for r in top],
        "series": [
            {"nom": "Ponctualité rapports (%)", "valeurs": [r.taux_ponctualite_rapports for r in top]},
            {"nom": "Présence à l'heure (%)", "valeurs": [r.taux_presence_a_temps for r in top]},
        ],
    }


async def graphique_presences_avs(avs_id: str, jours: int) -> dict | None:
    """Vue "ligne" simple : présent/retard/absent jour par jour pour un
    AVS donné — utile pour un coordonnateur qui suit une seule personne,
    ou pour un AVS qui consulte son propre historique."""
    presences = await donnees.recuperer_presences_avs(avs_id, jours)
    if not presences:
        return None

    presences = sorted(presences, key=lambda p: p["date"])
    valeur_statut = {"present": 2, "retard": 1, "absent": 0}
    return {
        "type": "line",
        "titre": f"Assiduité — {jours} derniers jours",
        "labels": [p["date"].date().isoformat() for p in presences],
        "series": [
            {
                "nom": "Statut (0=absent, 1=retard, 2=présent)",
                "valeurs": [valeur_statut.get(p.get("statut"), 0) for p in presences],
            }
        ],
    }
