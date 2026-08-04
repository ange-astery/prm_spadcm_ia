"""
Point d'entrée du service IA (prm-spadcm-ia). Voir README.md pour la
création, l'installation et la configuration complètes.

Lancement local : uvicorn app.main:app --reload --port 8010
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError

from app.api.routes import alertes, chat, evolution, health, performance, recherche, resume
from app.core.config import get_settings
from app.db.mongo import fermer_connexion, get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prm-spadcm-ia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Ouvre la connexion Mongo au démarrage (motor la garde en pool ensuite)
    # et vérifie qu'elle répond, pour échouer vite si MONGO_URI est faux.
    # Borné à 6s (serverSelectionTimeoutMS=5000 côté client, voir
    # app/db/mongo.py) : le service démarre quand même si Mongo est
    # injoignable au boot, mais chaque appel à un endpoint /ia/* échouera
    # explicitement tant que la base n'est pas accessible.
    try:
        await asyncio.wait_for(get_client().admin.command("ping"), timeout=6.0)
        logger.info("Connexion MongoDB (lecture seule) OK — base: %s", settings.mongo_db_name)
    except Exception:  # noqa: BLE001
        logger.warning("MongoDB injoignable au démarrage (MONGO_URI correct ? réseau ouvert vers Atlas ?)")

    if not settings.ia_service_token:
        logger.warning("IA_SERVICE_TOKEN non défini : toutes les routes /ia/* renverront 503.")
    cle_attendue = {
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
        "ollama": "ok",  # pas de clé nécessaire
    }.get(settings.llm_provider.lower().strip())
    if not cle_attendue:
        logger.warning(
            "Clé API manquante pour LLM_PROVIDER=%s : /ia/chat et /ia/resume-rapports renverront 503.",
            settings.llm_provider,
        )

    yield

    fermer_connexion()


app = FastAPI(
    title="prm-spadcm-ia",
    description=(
        "Service IA du PRM SPAD Cameroun : chatbot augmenté par les "
        "données du projet, résumés de rapports, analyse d'évolution "
        "santé, performance des AVS, recherche sémantique et alertes "
        "intelligentes. Appelé exclusivement par prm_spadcm_backend."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Ce service n'est jamais appelé depuis un navigateur/app directement, donc
# pas besoin d'une politique CORS permissive côté public ; on restreint par
# défaut et on documente comment l'ouvrir si un jour un usage le justifie.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(resume.router)
app.include_router(evolution.router)
app.include_router(performance.router)
app.include_router(recherche.router)
app.include_router(alertes.router)


@app.exception_handler(ValueError)
async def gerer_erreur_valeur(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PyMongoError)
async def gerer_erreur_mongo(request: Request, exc: PyMongoError):
    # Ne jamais renvoyer la stack trace brute (fuite d'infos sur l'URI/topologie
    # Mongo) : un message générique suffit, le détail reste dans les logs serveur.
    logger.error("Erreur MongoDB sur %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Base de données indisponible (lecture des données du projet impossible pour l'instant)."},
    )


@app.exception_handler(Exception)
async def gerer_erreur_inattendue(request: Request, exc: Exception):
    logger.exception("Erreur inattendue sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du service IA."})
