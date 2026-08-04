"""
Point d'entrée unique pour (ré-)entraîner les modèles du service IA.
Pensé pour un job planifié (cron / GitHub Actions) hors Vercel — voir
README.md section 11.5.6 : Vercel sert l'inférence, pas l'entraînement.

Usage :
    python -m app.training.cli --model all
    python -m app.training.cli --model anomalies_vitaux --synthetique
    python -m app.training.cli --model performance_avs
"""

from __future__ import annotations

import argparse
import sys

from app.training.entrainer_anomalies_vitaux import entrainer as entrainer_anomalies
from app.training.entrainer_performance_avs import entrainer as entrainer_performance

ENTRAINEURS = {
    "anomalies_vitaux": entrainer_anomalies,
    "performance_avs": entrainer_performance,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=[*ENTRAINEURS.keys(), "all"], default="all",
        help="Quel modèle (ré-)entraîner (défaut: all)",
    )
    parser.add_argument(
        "--synthetique", action="store_true",
        help="Force l'usage de données synthétiques même si assez de données réelles existent (utile pour tester le pipeline)",
    )
    args = parser.parse_args()

    cibles = list(ENTRAINEURS.keys()) if args.model == "all" else [args.model]
    echecs = []
    for nom in cibles:
        print(f"\n=== Entraînement : {nom} ===")
        try:
            ENTRAINEURS[nom](forcer_synthetique=args.synthetique)
        except ValueError as exc:
            print(f"  -> ignoré : {exc}")
            echecs.append(nom)

    if echecs:
        print(f"\n{len(echecs)}/{len(cibles)} modèle(s) non entraîné(s) faute de données : {', '.join(echecs)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
