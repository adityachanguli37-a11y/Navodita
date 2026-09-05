import json
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional


from calculator import process_calculator_query
from knowledge_engine import KnowledgeEngine
from nlp_utils import preprocess_text
from gemini_service import GeminiService
from universal_qa import answer_any_question

INTENTS_FILE = "data/intents.json"
MODEL_FILE = "models/chatbot_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
FALLBACK_TEXT = "I'm sorry, I wasn't able to find a good answer to that. Please try rephrasing your question."

# Keywords indicating informational, educational, or general technical queries
INFORMATIONAL_PREFIXES = (
    "what is", "what are", "what was", "what were", "what would", "what if", "what does",
    "why is", "why do", "why does", "why did", "why are", "why does the",
    "how is", "how do", "how does", "how did", "how can", "how to", "how would", "how do",
    "who is", "who was", "who were", "who invented", "who created",
    "where is", "where was", "where are",
    "when did", "when was", "when is",
    "explain", "describe", "define", "write", "code", "program", "implement",
    "difference between", "compare", "tell me about", "tell me an", "tell me a", "give me",
    "ideas for", "teach me", "can you explain", "could you explain"
)

# Common words that signal a conversational query rather than an external general question
CONVERSATIONAL_MARKERS = {
    "hello", "hi", "hey", "greetings", "good morning", "good evening", "good afternoon",
    "thanks", "thank you", "thx", "appreciate",
    "bye", "goodbye", "see you", "farewell", "later"
}

# Bot self-referential keywords
BOT_SELF_KEYWORDS = {"you", "your", "nexa", "nexaai", "operator", "bot", "assistant"}


def is_informational_query(text: str) -> bool:
    """
    Detect whether a query is seeking external information, explanation, or general knowledge.
    Informational queries must NEVER be hijacked by the ML intent classifier.
    """
    cleaned = text.strip().lower()
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", cleaned))

    # Explicit questions about the bot itself are permitted to match conversational bot intents
    if (tokens & BOT_SELF_KEYWORDS) and any(w in cleaned for w in ("who", "what are you", "your name", "your hours", "contact")):
        return False

    # Check leading informational prefixes
    for prefix in INFORMATIONAL_PREFIXES:
        if cleaned.startswith(prefix) or f" {prefix} " in f" {cleaned} ":
            return True

    # Check for programming / math / explanation keywords
    info_keywords = {
        "code", "python", "algorithm", "function", "program", "calculate",
        "solve", "why", "how", "what", "who", "where", "explain", "describe",
        "difference", "compare", "reverse", "quantum", "blockchain", "engine",
        "dream", "dreams", "star", "space", "earth", "gps", "wifi", "tcp", "udp"
    }
    if tokens & info_keywords:
        return True

    return False


def is_conversational_intent_candidate(text: str) -> bool:
    """
    Check if a query is strictly a conversational interaction (greeting, thanks, goodbye, bot info).
    Only these queries are permitted to be considered by the intent model.
    """
    cleaned = text.strip().lower()
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", cleaned))

    # If it is an informational query on an external subject, reject immediately
    if is_informational_query(text):
        return False

    # Greetings, thanks, goodbyes
    if any(m in cleaned for m in CONVERSATIONAL_MARKERS):
        return True

    # Bot self-inquiry (e.g. "what are your hours", "contact you", "who are you")
    if tokens & BOT_SELF_KEYWORDS:
        return True

    return False


class HybridChatbot:
    def __init__(
        self,
        intents_path: str = INTENTS_FILE,
        model_path: str = MODEL_FILE,
        vectorizer_path: str = VECTORIZER_FILE,
        confidence_threshold: float = 0.65
    ):
        self.intents_path = intents_path
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.confidence_threshold = confidence_threshold

        self.intents_map: Dict[str, List[str]] = {}
        self.classifier = None
        self.vectorizer = None
        self.knowledge_engine = KnowledgeEngine()
        self.gemini_service = GeminiService()

        self._load_resources()

    def _load_resources(self):
        """Load intents and serialized ML model / vectorizer."""
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("intents", []):
                    self.intents_map[item["tag"]] = item.get("responses", [])

        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.classifier = pickle.load(f)
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
            except Exception:
                self.classifier = None
                self.vectorizer = None

    def get_intent_prediction(self, text: str) -> Dict[str, Any]:
        """Classify input text using TF-IDF + Logistic Regression."""
        if not self.classifier or not self.vectorizer:
            return {"intent": "unknown", "confidence": 0.0}

        cleaned = preprocess_text(text)
        if not cleaned:
            return {"intent": "unknown", "confidence": 0.0}

        vec = self.vectorizer.transform([cleaned])
        probas = self.classifier.predict_proba(vec)[0]
        max_idx = int(probas.argmax())
        intent_tag = self.classifier.classes_[max_idx]
        conf = float(probas[max_idx])
        return {"intent": str(intent_tag), "confidence": round(conf, 2)}

    def validate_intent_match(self, query: str, intent_tag: str) -> bool:
        """
        Validate that the predicted intent ACTUALLY corresponds to the user's question.
        Rejects predictions if an intent is predicted for an unrelated question.
        """
        q_lower = query.lower()

        # Permitted intent tags and their strict required token anchors
        if intent_tag == "greeting":
            return any(w in q_lower for w in ("hello", "hi", "hey", "morning", "evening", "afternoon", "greetings"))
        if intent_tag == "goodbye":
            return any(w in q_lower for w in ("bye", "goodbye", "see you", "later", "farewell"))
        if intent_tag == "thanks":
            return any(w in q_lower for w in ("thank", "thanks", "thx", "appreciate"))
        if intent_tag == "hours":
            return any(w in q_lower for w in ("hour", "open", "timing", "schedule", "time"))
        if intent_tag == "contact":
            return any(w in q_lower for w in ("contact", "email", "phone", "reach", "call", "support"))
        if intent_tag == "location":
            return any(w in q_lower for w in ("where", "office", "headquarter", "address", "base", "located"))
        if intent_tag == "about_bot":
            return any(w in q_lower for w in ("who are you", "what are you", "your name", "nexa", "about you"))

        # Any general intent (like python, courses, etc.) must NEVER be answered via intent canned responses
        return False

    def respond(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Main Decision Pipeline:
        1. Input Validation
        2. Safe Mathematical Calculator (exact arithmetic, powers, sqrt)
        3. Simple Conversational Intent Guard (greetings, thanks, bye, bot contact)
        4. Primary Answering Engine: Ollama Local LLM (Qwen / Llama)
        5. High-Precision Local Knowledge Base (fallback for exact topic matches)
        6. Universal Open QA (Wikipedia encyclopedic lookup)
        7. Transparent Graceful Fallback
        """
        if not message or not message.strip():
            return {
                "response": "Please enter a message or question so I can assist you.",
                "answer": "Please enter a message or question so I can assist you.",
                "intent": "empty_input",
                "confidence": 1.0,
                "source": "intent_model",
                "success": True
            }

        user_query = message.strip()

        # ----------------------------------------------------
        # Layer 1: Safe AST Mathematical Calculator
        # ----------------------------------------------------
        calc_result = process_calculator_query(user_query)
        if calc_result:
            return {
                "response": f"Calculation result: {calc_result}",
                "answer": f"Calculation result: {calc_result}",
                "intent": "calculator",
                "confidence": 1.0,
                "source": "calculator",
                "success": True
            }

        # ----------------------------------------------------
        # Layer 2: Simple Conversational Intent Guard
        # Only simple greetings, thanks, goodbyes, and direct bot inquiries.
        # NEVER used for general knowledge, technical questions, or external topics.
        # ----------------------------------------------------
        if is_conversational_intent_candidate(user_query):
            intent_data = self.get_intent_prediction(user_query)
            intent_tag = intent_data["intent"]
            confidence = intent_data["confidence"]

            if confidence >= self.confidence_threshold and self.validate_intent_match(user_query, intent_tag):
                responses = self.intents_map.get(intent_tag, [])
                if responses:
                    selected_response = random.choice(responses)
                    return {
                        "response": selected_response,
                        "answer": selected_response,
                        "intent": intent_tag,
                        "confidence": confidence,
                        "source": "intent_model",
                        "success": True
                    }

        # ----------------------------------------------------
        # Layer 3: General Intelligence — Gemini API
        # ----------------------------------------------------
        if self.gemini_service.is_available():
            gemini_answer = self.gemini_service.generate_response(user_query, history=history)
            if gemini_answer and len(gemini_answer.strip()) > 0:
                return {
                    "response": gemini_answer,
                    "answer": gemini_answer,
                    "intent": "general_question",
                    "confidence": 0.97,
                    "source": "gemini",
                    "success": True
                }

        # ----------------------------------------------------
        # Layer 4: Local Curated Knowledge Base (fallback for exact topic matches)
        # ----------------------------------------------------
        kb_result = self.knowledge_engine.query(user_query, history=history)
        if kb_result:
            return {
                "response": kb_result["response"],
                "answer": kb_result["response"],
                "intent": "general_question",
                "confidence": kb_result["confidence"],
                "source": "knowledge_base",
                "title": kb_result.get("title"),
                "success": True
            }

        # ----------------------------------------------------
        # Layer 5: Universal Open Knowledge (Strict encyclopedic match)
        # ----------------------------------------------------
        universal_res = answer_any_question(user_query)
        if universal_res:
            return {
                "response": universal_res["response"],
                "answer": universal_res["response"],
                "intent": "general_question",
                "confidence": universal_res["confidence"],
                "source": "knowledge_base",
                "title": universal_res.get("title"),
                "success": True
            }

        # ----------------------------------------------------
        # Layer 7: Truthful, Transparent Fallback
        # ----------------------------------------------------
        return {
            "response": FALLBACK_TEXT,
            "answer": FALLBACK_TEXT,
            "intent": "unknown",
            "confidence": 0.0,
            "source": "fallback",
            "success": False
        }
