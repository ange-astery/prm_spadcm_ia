"""
Client LLM du service IA — POINT UNIQUE à modifier pour changer de
fournisseur (voir README.md, section "Changer de fournisseur LLM").

Tout le reste de l'application (app/services/chat.py, resume.py) appelle
uniquement `generer_texte(system_prompt, messages, max_tokens)` et ne
sait jamais quel fournisseur répond derrière. Changer de fournisseur =
changer `LLM_PROVIDER` (+ la clé correspondante) dans `.env`, sans
toucher au reste du code.

Fournisseurs supportés :
- "gemini"    : Google Gemini — gratuit via Google AI Studio, sans CB (recommandé pour démarrer)
- "groq"      : Groq — gratuit, très rapide, modèles open-source (Llama, Mixtral)
- "ollama"    : 100% local, gratuit et illimité (développement uniquement)
- "anthropic" : Claude — payant, conservé en option
"""

import httpx
from fastapi import HTTPException, status

from app.core.config import get_settings

# `messages` suit TOUJOURS le même format en entrée de `generer_texte`,
# quel que soit le fournisseur choisi ensuite :
#   [{"role": "user"|"assistant", "content": "..."}]


async def generer_texte(system_prompt: str, messages: list[dict], max_tokens: int = 1024) -> str:
    settings = get_settings()
    fournisseur = settings.llm_provider.lower().strip()

    if fournisseur == "gemini":
        return await _generer_gemini(system_prompt, messages, max_tokens)
    if fournisseur == "groq":
        return await _generer_groq(system_prompt, messages, max_tokens)
    if fournisseur == "ollama":
        return await _generer_ollama(system_prompt, messages, max_tokens)
    if fournisseur == "anthropic":
        return await _generer_anthropic(system_prompt, messages, max_tokens)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"LLM_PROVIDER inconnu: '{settings.llm_provider}' (attendu: gemini, groq, ollama, anthropic).",
    )


# --------------------------------------------------------------------------- #
# Google Gemini — https://ai.google.dev
# --------------------------------------------------------------------------- #

async def _generer_gemini(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY non configurée côté service IA.",
        )

    # Gemini utilise "model" au lieu de "assistant" pour le rôle du LLM,
    # et sépare le prompt système dans `system_instruction`.
    contenus = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        reponse = await client.post(
            url,
            headers={"content-type": "application/json", "x-goog-api-key": settings.gemini_api_key},
            json={
                "contents": contenus,
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
        )

    if reponse.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini momentanément indisponible ({reponse.status_code}): {reponse.text[:300]}",
        )

    donnees = reponse.json()
    candidats = donnees.get("candidates") or []
    if not candidats:
        return "Désolé, je n'ai pas pu formuler de réponse."

    # Chaque "part" peut contenir soit du texte ("text"), soit une trace de
    # raisonnement interne ("thoughtSignature", modèles "thinking") : on ne
    # garde QUE les parts qui ont une clé "text".
    parts = candidats[0].get("content", {}).get("parts", [])
    texte = "\n".join(p["text"] for p in parts if "text" in p).strip()
    return texte or "Désolé, je n'ai pas pu formuler de réponse."


# --------------------------------------------------------------------------- #
# Groq — API compatible OpenAI, https://console.groq.com
# --------------------------------------------------------------------------- #

async def _generer_groq(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY non configurée côté service IA.",
        )

    payload_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        reponse = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {settings.groq_api_key}",
            },
            json={
                "model": settings.groq_model,
                "messages": payload_messages,
                "max_tokens": max_tokens,
            },
        )

    if reponse.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq momentanément indisponible ({reponse.status_code}): {reponse.text[:300]}",
        )

    donnees = reponse.json()
    contenu = donnees["choices"][0]["message"]["content"]
    return (contenu or "").strip() or "Désolé, je n'ai pas pu formuler de réponse."


# --------------------------------------------------------------------------- #
# Ollama — 100% local, https://ollama.com
# --------------------------------------------------------------------------- #

async def _generer_ollama(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
    settings = get_settings()

    payload_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # local, peut être plus lent selon la machine
            reponse = await client.post(
                url,
                json={
                    "model": settings.ollama_model,
                    "messages": payload_messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Impossible de joindre Ollama en local "
                f"({settings.ollama_base_url}). Ollama est-il lancé (`ollama serve`) "
                f"et le modèle '{settings.ollama_model}' bien téléchargé (`ollama pull {settings.ollama_model}`) ?"
            ),
        ) from exc

    if reponse.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama momentanément indisponible ({reponse.status_code}): {reponse.text[:300]}",
        )

    donnees = reponse.json()
    contenu = donnees.get("message", {}).get("content", "")
    return contenu.strip() or "Désolé, je n'ai pas pu formuler de réponse."


# --------------------------------------------------------------------------- #
# Anthropic Claude — payant, conservé en option
# --------------------------------------------------------------------------- #

async def _generer_anthropic(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ANTHROPIC_API_KEY non configurée côté service IA.",
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        reponse = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "content-type": "application/json",
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": settings.anthropic_model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
            },
        )

    if reponse.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Anthropic momentanément indisponible ({reponse.status_code}): {reponse.text[:300]}",
        )

    donnees = reponse.json()
    blocs_texte = [b["text"] for b in donnees.get("content", []) if b.get("type") == "text"]
    return "\n".join(blocs_texte).strip() or "Désolé, je n'ai pas pu formuler de réponse."
