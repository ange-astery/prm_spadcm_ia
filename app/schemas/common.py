from typing import Literal

from pydantic import BaseModel, Field

RoleUtilisateur = Literal["patient", "avs", "medecin", "coordonnateur", "administrateur"]


class ContexteUtilisateur(BaseModel):
    """
    Le backend Node connaît déjà l'utilisateur authentifié (req.user) : il
    transmet ici le strict nécessaire pour que ce service adapte sa
    réponse, sans jamais recevoir le token JWT de l'utilisateur final.
    """

    id: str
    role: RoleUtilisateur
    prenom: str
    nom: str | None = None
