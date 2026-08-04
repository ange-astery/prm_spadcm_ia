"""
Connexion à la même base MongoDB Atlas que prm_spadcm_backend, en LECTURE
SEULE (voir README.md : créer un utilisateur Atlas dédié avec le rôle
"read"). Ce module ne fournit volontairement aucune fonction d'écriture :
si un jour un besoin d'écriture apparaît, il doit passer par un callback
HTTP vers le backend Node (voir app/services/callback_backend.py), jamais
par une écriture Mongo directe depuis ce service.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        # serverSelectionTimeoutMS court : en cas de MONGO_URI invalide/injoignable,
        # on échoue vite (voir app/main.py) plutôt que de bloquer le démarrage
        # du service sur le timeout par défaut du driver (30s).
        _client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return get_client()[settings.mongo_db_name]


def fermer_connexion() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
