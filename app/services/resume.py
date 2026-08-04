from datetime import date, timedelta

from app.schemas.rapports import ReponseResume, RequeteResume
from app.services import donnees
from app.services.callback_backend import enregistrer_resume_rapport
from app.services.llm import generer_texte

SYSTEM_PROMPT_RESUME = (
    "Tu es un assistant qui résume, pour un professionnel de santé ou une "
    "famille, une série de rapports journaliers de soins à domicile "
    "(SPAD Cameroun). Résume en français, en 4 à 8 phrases maximum : "
    "état général, évolution notable, points d'attention. Ne pose jamais "
    "de diagnostic. Si les données sont insuffisantes, dis-le simplement."
)


def _formater_rapport_pour_prompt(rapport: dict) -> str:
    date_str = rapport.get("date")
    date_str = date_str.date().isoformat() if hasattr(date_str, "date") else str(date_str)
    morceaux = [f"Date: {date_str}"]

    if rapport.get("rapportPatient"):
        morceaux.append(f"Observation générale: {rapport['rapportPatient']}")
    if rapport.get("plainte"):
        morceaux.append(f"Plainte du patient: {rapport['plainte']}")
    if rapport.get("observations"):
        morceaux.append(f"Observations AVS: {rapport['observations']}")
    if rapport.get("conclusion"):
        morceaux.append(f"Conclusion: {rapport['conclusion']}")

    for releve in rapport.get("parametresVitaux", []) or []:
        details = ", ".join(
            f"{cle}={valeur}"
            for cle, valeur in releve.items()
            if cle not in ("moment", "notes") and valeur not in (None, "")
        )
        morceaux.append(f"Constantes ({releve.get('moment', '?')}): {details}")

    return " | ".join(morceaux)


async def generer_resume(requete: RequeteResume) -> ReponseResume:
    date_fin = requete.date_fin or date.today()
    date_debut = requete.date_debut or (date_fin - timedelta(days=7))

    rapports = await donnees.recuperer_rapports_patient(
        requete.patient_id, date_debut=date_debut, date_fin=date_fin
    )

    if not rapports:
        resume = "Aucun rapport journalier disponible sur cette période."
    else:
        texte_rapports = "\n".join(_formater_rapport_pour_prompt(r) for r in reversed(rapports))
        resume = await generer_texte(
            system_prompt=SYSTEM_PROMPT_RESUME,
            messages=[
                {
                    "role": "user",
                    "content": f"Voici les rapports journaliers à résumer :\n\n{texte_rapports}",
                }
            ],
            max_tokens=400,
        )

    if requete.ecrire_dans_rapport_id:
        await enregistrer_resume_rapport(requete.ecrire_dans_rapport_id, resume)

    return ReponseResume(
        patient_id=requete.patient_id,
        periode=f"{date_debut.isoformat()} → {date_fin.isoformat()}",
        resume=resume,
        nombre_rapports_analyses=len(rapports),
    )
