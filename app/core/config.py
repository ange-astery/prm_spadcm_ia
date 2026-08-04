"""
Configuration centralisée du service IA, chargée depuis les variables
d'environnement (voir .env.example). Utilise pydantic-settings pour
avoir des valeurs typées et validées au démarrage plutôt que des
os.getenv() éparpillés dans le code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Serveur
    port: int = 8010
    env: str = "development"

    # MongoDB (lecture seule)
    mongo_uri: str = "mongodb://localhost:27017/prm_spad_test"
    mongo_db_name: str = "prm_spad_test"

    # Sécurité interservice
    ia_service_token: str = ""
    ia_service_url: str = "http://localhost:8010"

    # LLM — fournisseur choisi via LLM_PROVIDER, un seul à la fois.
    # "anthropic" | "gemini" | "groq" | "ollama"
    llm_provider: str = "gemini"

    # Anthropic (payant — conservé en option, ex. si vous voulez comparer)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Google Gemini (gratuit via Google AI Studio, sans carte bancaire)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"

    # Groq (gratuit, très rapide, modèles open-source type Llama/Mixtral)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Ollama (100% local, gratuit et illimité, pour le développement)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Callback vers le backend Node
    backend_url: str = "http://localhost:4000"
    backend_callback_token: str = ""

    # RAG
    rag_top_k: int = 6

    @property
    def est_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    """Singleton simple : évite de relire/reparser le .env à chaque requête."""
    return Settings()
