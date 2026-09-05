import os
import uuid
from typing import Dict, List
from flask import Flask, jsonify, render_template, request
from chatbot import HybridChatbot

# Load .env file for local development (no-op if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; env vars must be set manually

# Ensure NLTK corpora are present (required on cloud deployments like Render)
import nltk
for _corpus in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"):
    try:
        nltk.download(_corpus, quiet=True)
    except Exception:
        pass

app = Flask(__name__)
bot = HybridChatbot()

# In-memory store for bounded conversation sessions keyed by conversation_id
CONVERSATION_SESSIONS: Dict[str, List[Dict[str, str]]] = {}
MAX_SESSION_HISTORY = 10


@app.route("/")
def index():
    """Render the primary NexaAI conversational user interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main Chat API Endpoint.
    Expects JSON: { "message": "...", "conversation_id": "optional", "history": [...] }
    Returns JSON: { "success": true, "answer": "...", "source": "...", "response": "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "")
        conv_id = data.get("conversation_id")
        client_history = data.get("history")

        # Resolve or create conversation session
        if not conv_id:
            conv_id = str(uuid.uuid4())

        if conv_id not in CONVERSATION_SESSIONS:
            CONVERSATION_SESSIONS[conv_id] = []

        # Use server-side session history; initialize from client history if server session is empty
        session_history = CONVERSATION_SESSIONS[conv_id]
        if not session_history and client_history and isinstance(client_history, list):
            clean_client_history = []
            for item in client_history[-MAX_SESSION_HISTORY:]:
                if isinstance(item, dict) and item.get("role") and item.get("content"):
                    # Exclude an accidental trailing duplicate of current query
                    if item.get("content", "").strip() != user_message.strip():
                        clean_client_history.append({
                            "role": item["role"],
                            "content": str(item["content"]).strip()
                        })
            session_history = clean_client_history
            CONVERSATION_SESSIONS[conv_id] = session_history

        # Filter session history so current question isn't in prior history
        prior_history = [
            turn for turn in session_history
            if not (turn.get("role") == "user" and turn.get("content", "").strip() == user_message.strip())
        ]

        # Execute hybrid AI pipeline
        result = bot.respond(user_message, history=prior_history)

        answer_text = result.get("answer") or result.get("response", "")
        source_val = result.get("source", "fallback")
        is_success = result.get("success", source_val != "fallback")

        # Update conversation session history
        if user_message.strip() and answer_text:
            CONVERSATION_SESSIONS[conv_id].append({"role": "user", "content": user_message.strip()})
            CONVERSATION_SESSIONS[conv_id].append({"role": "assistant", "content": answer_text})
            # Keep bounded
            if len(CONVERSATION_SESSIONS[conv_id]) > MAX_SESSION_HISTORY * 2:
                CONVERSATION_SESSIONS[conv_id] = CONVERSATION_SESSIONS[conv_id][-(MAX_SESSION_HISTORY * 2):]

        return jsonify({
            "success": is_success,
            "answer": answer_text,
            "response": answer_text,
            "source": source_val,
            "confidence": result.get("confidence", 0.0),
            "conversation_id": conv_id
        })

    except Exception:
        # Never leak internal errors or stack traces
        return jsonify({
            "success": False,
            "answer": "An unexpected service error occurred. Please try again shortly.",
            "response": "An unexpected service error occurred. Please try again shortly.",
            "source": "fallback",
            "confidence": 0.0
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck endpoint reporting subsystem statuses."""
    return jsonify({
        "status": "healthy",
        "service": "NexaAI Universal AI Chatbot",
        "gemini_available": bot.gemini_service.is_available(),
        "has_intent_model": bot.classifier is not None,
        "has_knowledge_engine": len(bot.knowledge_engine.documents) > 0,
        "knowledge_topics_count": len(bot.knowledge_engine.documents)
    })


@app.route("/topics", methods=["GET"])
def get_topics():
    """List available knowledge base topics."""
    topics = [
        {"topic": doc.get("topic"), "title": doc.get("title")}
        for doc in bot.knowledge_engine.documents
    ]
    return jsonify({"topics": topics})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
