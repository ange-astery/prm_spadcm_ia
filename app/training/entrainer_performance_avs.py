"""
Entraîne un modèle de régression prédisant la note moyenne
d'appréciation d'un AVS à partir de ses signaux de ponctualité, en
complément (jamais en remplacement) du score par règles de
app/services/performance_avs.py — voir README.md section 11.5.3.

Modèle : GradientBoostingRegressor (scikit-learn). Produit l'étude de
performance associée (dispersion prédiction/réalité, distribution des
résidus, importance des variables, courbe d'apprentissage).

Usage :
    python -m app.training.entrainer_performance_avs
    python -m app.training.entrainer_performance_avs --synthetique
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split

from app.training import curves
from app.training.donnees_entrainement import construire_dataset_performance_avs
from app.training.registry import sauvegarder_version

NOM_MODELE = "performance_avs"


def entrainer(forcer_synthetique: bool = False) -> str:
    jeu = construire_dataset_performance_avs(forcer_synthetique=forcer_synthetique)
    X, y = jeu.X, jeu.y

    if len(X) < 20:
        raise ValueError(
            f"Seulement {len(X)} exemples disponibles : trop peu pour entraîner un modèle "
            "de régression fiable. Relancez avec --synthetique en attendant plus d'historique réel."
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    modele = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    modele.fit(X_train, y_train)

    y_predit = modele.predict(X_test)

    metriques = {
        "mae": round(mean_absolute_error(y_test, y_predit), 4),
        "r2": round(r2_score(y_test, y_predit), 4),
        "n_exemples_train": len(X_train),
        "n_exemples_test": len(X_test),
    }

    with tempfile.TemporaryDirectory() as tmp:
        dossier_courbes = Path(tmp)
        chemins_courbes = [
            curves.dispersion_predictions(list(y_test), list(y_predit), dossier_courbes),
            curves.distribution_residus(list(y_test), list(y_predit), dossier_courbes),
            curves.importance_variables(jeu.colonnes, list(modele.feature_importances_), dossier_courbes),
        ]

        tailles, score_train, score_val = learning_curve(
            GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42),
            X, y, cv=5, train_sizes=[0.3, 0.5, 0.7, 0.85, 1.0], scoring="r2", random_state=42,
        )[0:3]
        chemins_courbes.append(
            curves.courbe_apprentissage(
                list(tailles), list(score_train.mean(axis=1)), list(score_val.mean(axis=1)), dossier_courbes,
                titre="Courbe d'apprentissage (R²)",
            )
        )

        version = sauvegarder_version(
            NOM_MODELE, modele, metriques, chemins_courbes,
            meta={"source_donnees": jeu.source, "colonnes": jeu.colonnes},
        )

    print(f"[{NOM_MODELE}] version {version} entraînée sur données {jeu.source} — {metriques}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetique", action="store_true", help="Force l'usage de données synthétiques")
    args = parser.parse_args()
    entrainer(forcer_synthetique=args.synthetique)
