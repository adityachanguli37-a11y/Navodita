"""
GeminiService — Cloud LLM backend using Google Gemini API.

Drop-in replacement for OllamaService when running in the cloud (e.g., Render).
Reads GEMINI_API_KEY from the environment. Falls back gracefully when the key
is absent so local development with Ollama is unaffected.

Uses the current `google-genai` SDK (google.genai).

Free tier limits (gemini-2.0-flash-lite):
    - 30 RPM, 1 500 req/day, 1M tokens/day
"""

import os
import warnings
from typing import List, Dict, Optional

# Suppress noisy AFC recommendation warning from google.genai
warnings.filterwarnings("ignore", category=FutureWarning, module="google.genai")

MAX_HISTORY_TURNS = 6

SYSTEM_INSTRUCTION = (
    "You are a highly capable general-purpose AI assistant. "
    "Answer the user question accurately and directly. "
    "Carefully determine what the user is asking before answering. "
    "Never answer a different question. "
    "Use conversation history to resolve pronoun references. "
    "If the user requests code, provide relevant code. "
    "If real-time information is needed that you cannot verify, say so honestly. "
    "Keep answers clear, accurate, and appropriately detailed."
)

_GEMINI_MODEL = "gemini-3.5-flash-lite"


class GeminiService:
    """Cloud LLM service backed by Google Gemini. Requires GEMINI_API_KEY env var."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = None
        self._available: bool = False

        if self._api_key:
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore
                self._genai = genai
                self._types = types
                self._client = genai.Client(api_key=self._api_key)
                self._available = True
            except Exception:
                self._available = False

    def is_available(self) -> bool:
        return self._available

    def _build_contents(
        self, user_message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List:
        """
        Convert conversation history to google.genai `contents` format.
        Gemini uses 'user' and 'model' roles (not 'assistant').
        """
        contents = []
        if history and isinstance(history, list):
            for msg in history[-MAX_HISTORY_TURNS:]:
                role = msg.get("role", "")
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    self._types.Content(
                        role=gemini_role,
                        parts=[self._types.Part(text=content)]
                    )
                )

        clean_msg = user_message.strip()
        # Avoid duplicate last user turn
        if not contents or contents[-1].role != "user" or contents[-1].parts[0].text != clean_msg:
            contents.append(
                self._types.Content(
                    role="user",
                    parts=[self._types.Part(text=clean_msg)]
                )
            )
        return contents

    def generate_response(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Generate a response from Gemini.
        Returns None if the service is unavailable or the call fails.
        """
        if not self._available or not self._client:
            return None

        clean_message = user_message.strip()
        if not clean_message:
            return None

        try:
            contents = self._build_contents(clean_message, history=history)
            response = self._client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    max_output_tokens=512,
                )
            )
            text = response.text.strip() if response and response.text else ""
            return text if text else None
        except Exception:
            return None
