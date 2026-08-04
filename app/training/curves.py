"""
Générateurs de courbes d'étude de performance pour l'entraînement des
modèles du service IA (voir README.md, section "Entraîner ses propres
modèles"). Chaque fonction sauvegarde un PNG dans le dossier fourni et
renvoie son chemin, pour que les scripts d'entraînement puissent lister
ce qu'ils ont produit dans metrics.json.

Backend "Agg" forcé : ces scripts tournent en CLI/job planifié, jamais
dans un contexte avec affichage — voir README.md, section 11.6 (Vercel
sert l'inférence, pas l'entraînement).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def _sauvegarder(fig, dossier: Path, nom_fichier: str) -> str:
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / nom_fichier
    fig.savefig(chemin, bbox_inches="tight")
    plt.close(fig)
    return str(chemin)


def courbe_apprentissage(tailles: list[int], score_train: list[float], score_val: list[float], dossier: Path, titre: str = "Courbe d'apprentissage") -> str:
    """Score (ex. R² ou accuracy) en fonction de la taille du jeu
    d'entraînement — sert à détecter sous/sur-apprentissage et à juger
    si on a besoin de plus de données réelles avant de refaire confiance
    au modèle."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(tailles, score_train, marker="o", label="Entraînement")
    ax.plot(tailles, score_val, marker="o", label="Validation")
    ax.set_xlabel("Nombre d'exemples d'entraînement")
    ax.set_ylabel("Score")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    return _sauvegarder(fig, dossier, "courbe_apprentissage.png")


def matrice_confusion(y_vrai, y_predit, classes: list[str], dossier: Path, titre: str = "Matrice de confusion") -> str:
    from sklearn.metrics import confusion_matrix

    matrice = confusion_matrix(y_vrai, y_predit, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrice, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(titre)
    for i in range(matrice.shape[0]):
        for j in range(matrice.shape[1]):
            ax.text(j, i, str(matrice[i, j]), ha="center", va="center",
                     color="white" if matrice[i, j] > matrice.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _sauvegarder(fig, dossier, "matrice_confusion.png")


def courbe_roc(y_vrai, scores, dossier: Path, titre: str = "Courbe ROC") -> str:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_vrai, scores)
    aire = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {aire:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Hasard")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    return _sauvegarder(fig, dossier, "courbe_roc.png")


def courbe_precision_rappel(y_vrai, scores, dossier: Path, titre: str = "Précision / Rappel") -> str:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, rappel, _ = precision_recall_curve(y_vrai, scores)
    ap = average_precision_score(y_vrai, scores)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rappel, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    return _sauvegarder(fig, dossier, "courbe_precision_rappel.png")


def importance_variables(noms_variables: list[str], importances: list[float], dossier: Path, titre: str = "Importance des variables") -> str:
    ordre = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(5, 0.4 * len(noms_variables) + 1))
    ax.barh([noms_variables[i] for i in ordre], [importances[i] for i in ordre], color="#3B82A0")
    ax.set_xlabel("Importance")
    ax.set_title(titre)
    ax.grid(alpha=0.3, axis="x")
    return _sauvegarder(fig, dossier, "importance_variables.png")


def dispersion_predictions(y_vrai, y_predit, dossier: Path, titre: str = "Prédit vs réel") -> str:
    """Nuage de points prédiction/réalité pour un modèle de régression
    (ex. score de performance AVS) + droite idéale y=x en référence."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_vrai, y_predit, alpha=0.6, edgecolor="k", linewidth=0.3)
    bornes = [min(min(y_vrai), min(y_predit)), max(max(y_vrai), max(y_predit))]
    ax.plot(bornes, bornes, linestyle="--", color="grey", label="Prédiction parfaite")
    ax.set_xlabel("Valeur réelle")
    ax.set_ylabel("Valeur prédite")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    return _sauvegarder(fig, dossier, "dispersion_predictions.png")


def distribution_residus(y_vrai, y_predit, dossier: Path, titre: str = "Distribution des résidus") -> str:
    residus = np.array(y_predit) - np.array(y_vrai)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(residus, bins=min(20, max(5, len(residus) // 3)), color="#B0413E", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Erreur (prédit - réel)")
    ax.set_ylabel("Nombre d'exemples")
    ax.set_title(titre)
    ax.grid(alpha=0.3, axis="y")
    return _sauvegarder(fig, dossier, "distribution_residus.png")
