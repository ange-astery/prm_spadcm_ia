"""
Entraîne un classifieur (détection d'anomalie sur un relevé de
constantes vitales) et produit l'étude de performance associée (courbe
ROC, précision/rappel, matrice de confusion, importance des variables,
courbe d'apprentissage) — voir README.md section 11.5.3.

Modèle : RandomForestClassifier (scikit-learn). Choisi plutôt qu'un
réseau de neurones : peu de données au départ, résultats explicables
(importance des variables directement interprétable par un médecin
référent du projet), robuste sans réglage fin.

Usage :
    python -m app.training.entrainer_anomalies_vitaux
    python -m app.training.entrainer_anomalies_vitaux --synthetique
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import learning_curve, train_test_split

from app.training import curves
from app.training.donnees_entrainement import construire_dataset_anomalies_vitaux
from app.training.registry import sauvegarder_version

NOM_MODELE = "anomalies_vitaux"


def entrainer(forcer_synthetique: bool = False) -> str:
    jeu = construire_dataset_anomalies_vitaux(forcer_synthetique=forcer_synthetique)
    X, y = jeu.X.fillna(jeu.X.median(numeric_only=True)), jeu.y

    if y.nunique() < 2:
        raise ValueError(
            "Le jeu de données n'a qu'une seule classe (aucune anomalie détectée) : "
            "impossible d'entraîner/évaluer un classifieur binaire. Relancez avec "
            "--synthetique pour vérifier le pipeline en attendant plus de données réelles."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modele = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    modele.fit(X_train, y_train)

    y_predit = modele.predict(X_test)
    scores = modele.predict_proba(X_test)[:, 1]

    metriques = {
        "exactitude": round(accuracy_score(y_test, y_predit), 4),
        "precision": round(precision_score(y_test, y_predit, zero_division=0), 4),
        "rappel": round(recall_score(y_test, y_predit, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_predit, zero_division=0), 4),
        "n_exemples_train": len(X_train),
        "n_exemples_test": len(X_test),
        "taux_anomalies_dataset": round(float(y.mean()), 4),
    }

    with tempfile.TemporaryDirectory() as tmp:
        dossier_courbes = Path(tmp)
        chemins_courbes = []

        chemins_courbes.append(
            curves.matrice_confusion(y_test, y_predit, classes=["normal", "anomalie"], dossier=dossier_courbes)
        )
        # ROC/PR nécessitent les deux classes présentes dans y_test.
        if y_test.nunique() == 2:
            chemins_courbes.append(curves.courbe_roc(y_test, scores, dossier_courbes))
            chemins_courbes.append(curves.courbe_precision_rappel(y_test, scores, dossier_courbes))

        chemins_courbes.append(
            curves.importance_variables(jeu.colonnes, list(modele.feature_importances_), dossier_courbes)
        )

        try:
            tailles, score_train, score_val = learning_curve(
                RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
                X, y, cv=min(5, y.value_counts().min()), train_sizes=[0.3, 0.5, 0.7, 0.85, 1.0],
                scoring="f1", random_state=42,
            )[0:3]
            chemins_courbes.append(
                curves.courbe_apprentissage(
                    list(tailles), list(score_train.mean(axis=1)), list(score_val.mean(axis=1)), dossier_courbes,
                    titre="Courbe d'apprentissage (F1)",
                )
            )
        except ValueError:
            pass  # pas assez d'exemples par classe pour la validation croisée — on garde le reste du rapport

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
