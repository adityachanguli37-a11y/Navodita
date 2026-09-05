import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app import app
from chatbot import HybridChatbot, is_informational_query, is_conversational_intent_candidate
from calculator import evaluate_math_expression, process_calculator_query


@pytest.fixture(scope="module")
def bot():
    """Shared chatbot instance with Gemini disabled for fast unit tests."""
    b = HybridChatbot(confidence_threshold=0.55)
    # Disable Gemini for fast unit tests - mock it to always return None
    b.gemini_service = MagicMock()
    b.gemini_service.generate_response.return_value = None
    b.gemini_service.is_available.return_value = False
    return b


@pytest.fixture(scope="module")
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# ----------------------------------------------------
# 1. Calculator Tests
# ----------------------------------------------------
def test_calculator_basic_operations():
    assert process_calculator_query("25 + 75") == "25 + 75 = 100"
    assert process_calculator_query("100 / 4") == "100 / 4 = 25"
    assert process_calculator_query("12 * 8") == "12 * 8 = 96"
    assert process_calculator_query("50 - 17") == "50 - 17 = 33"
    assert process_calculator_query("100 / 5") == "100 / 5 = 20"
    assert process_calculator_query("What is 25 * 8?") == "25 * 8 = 200"


def test_calculator_powers_parentheses_and_sqrt():
    assert process_calculator_query("2 ** 5") == "2 ** 5 = 32"
    assert process_calculator_query("2 ** 10") == "2 ** 10 = 1024"
    assert process_calculator_query("(20 + 30) * 4") == "(20 + 30) * 4 = 200"
    assert process_calculator_query("sqrt(144)") == "sqrt(144) = 12"
    assert process_calculator_query("calculate sqrt(25)") == "sqrt(25) = 5"


def test_calculator_division_by_zero():
    res = process_calculator_query("100 / 0")
    assert "Division by zero" in res or "undefined" in res


def test_calculator_rejects_arbitrary_code():
    assert process_calculator_query("import os; os.system('dir')") is None
    assert process_calculator_query("__import__('os')") is None
    assert process_calculator_query("open('file.txt')") is None
    assert process_calculator_query("exec('x=1')") is None


def test_calculator_rejects_questions_about_math():
    assert process_calculator_query("Why does multiplication work?") is None
    assert process_calculator_query("Explain the Pythagorean theorem.") is None


# ----------------------------------------------------
# 2. Intent Hijacking Protection Tests
# ----------------------------------------------------
def test_informational_query_detection():
    assert is_informational_query("Explain quantum computing.") is True
    assert is_informational_query("Who was Albert Einstein?") is True
    assert is_informational_query("Explain blockchain in simple words.") is True
    assert is_informational_query("What is machine learning?") is True
    assert is_informational_query("Why do humans dream?") is True
    assert is_informational_query("Write a Python program to sort a list.") is True
    assert is_informational_query("How does a car engine work?") is True
    assert is_informational_query("Why does the Moon not fall onto Earth?") is True

    assert is_informational_query("Hello") is False
    assert is_informational_query("Goodbye") is False
    assert is_informational_query("Thank you") is False


def test_intent_candidate_filtering():
    assert is_conversational_intent_candidate("Hello") is True
    assert is_conversational_intent_candidate("Good morning") is True
    assert is_conversational_intent_candidate("Thanks a lot") is True
    assert is_conversational_intent_candidate("Goodbye") is True
    assert is_conversational_intent_candidate("What are your business hours?") is True

    assert is_conversational_intent_candidate("Explain blockchain.") is False
    assert is_conversational_intent_candidate("What is quantum computing?") is False
    assert is_conversational_intent_candidate("How does a car engine work?") is False
    assert is_conversational_intent_candidate("Why do humans dream?") is False


def test_intent_hijacking_blocked_for_arbitrary_questions(bot):
    """General questions must NEVER be answered by the intent model."""
    arbitrary_queries = [
        "Explain quantum computing.",
        "Explain blockchain in simple words.",
        "Why do humans dream?",
        "How does a car engine work?",
        "Why does the Moon not fall onto Earth?"
    ]
    for q in arbitrary_queries:
        res = bot.respond(q)
        assert res["source"] != "intent_model", f"Query '{q}' was hijacked by intent_model!"


def test_intent_hijacking_rejection_even_with_high_confidence(bot):
    """Even at 0.99 confidence, wrong intent must be rejected."""
    with patch.object(bot, "get_intent_prediction", return_value={"intent": "python", "confidence": 0.99}):
        res = bot.respond("Explain blockchain.")
        assert res["source"] != "intent_model"
        assert res["intent"] != "python"


# ----------------------------------------------------
# 3. Conversational Intent Tests
# ----------------------------------------------------
def test_intent_greeting(bot):
    res = bot.respond("Hello NexaAI")
    assert res["source"] == "intent_model"
    assert res["intent"] == "greeting"
    assert res["confidence"] >= 0.55


def test_intent_goodbye(bot):
    res = bot.respond("Goodbye, see you later!")
    assert res["source"] == "intent_model"
    assert res["intent"] == "goodbye"


def test_intent_thanks(bot):
    res = bot.respond("Thank you very much for your help")
    assert res["source"] == "intent_model"
    assert res["intent"] == "thanks"


def test_intent_hours_and_contact(bot):
    res_hours = bot.respond("What are your business hours?")
    assert res_hours["source"] == "intent_model"
    assert res_hours["intent"] == "hours"

    res_contact = bot.respond("How can I contact you?")
    assert res_contact["source"] == "intent_model"
    assert res_contact["intent"] == "contact"


# ----------------------------------------------------
# 4. Knowledge Retrieval Tests
# ----------------------------------------------------
def test_knowledge_machine_learning(bot):
    res = bot.respond("What is machine learning?")
    assert res["source"] in ("knowledge_base", "gemini", "fallback")
    assert "machine learning" in (res.get("answer") or res.get("response")).lower()


def test_knowledge_python(bot):
    res = bot.respond("Explain Python.")
    assert res["source"] in ("knowledge_base", "gemini", "fallback")
    assert "python" in (res.get("answer") or res.get("response")).lower()


def test_knowledge_rejects_unrelated_questions(bot):
    """Knowledge engine must not return loosely matched topics."""
    unrelated_queries = [
        "Explain black holes.",
        "How does a car engine work?",
        "Why do humans dream?",
        "What would happen if the Earth stopped rotating?",
        "Why does the Moon not fall onto Earth?"
    ]
    for q in unrelated_queries:
        kb_match = bot.knowledge_engine.query(q)
        assert kb_match is None, f"Query '{q}' matched KB topic '{kb_match.get('title') if kb_match else None}'!"


# ----------------------------------------------------
# 5. Contextual Follow-up Tests
# ----------------------------------------------------
def test_contextual_follow_up(bot):
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."}
    ]
    res = bot.respond("What are its advantages?", history=history)
    assert res["source"] in ("knowledge_base", "gemini", "fallback")


def test_topic_switching_does_not_pollute(bot):
    history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI."}
    ]
    res = bot.respond("How does SQL work?", history=history)
    text = (res.get("answer") or res.get("response")).lower()
    assert "sql" in text or "database" in text or res["source"] == "fallback"
    if res["source"] == "knowledge_base":
        title_lower = (res.get("title") or "").lower()
        assert "sql" in title_lower or "database" in title_lower


# ----------------------------------------------------
# 6. Boundary & Fallback Tests
# ----------------------------------------------------
def test_empty_message(bot):
    res = bot.respond("   ")
    assert res["intent"] == "empty_input"
    assert res["confidence"] == 1.0


def test_unknown_question_graceful_handling(bot):
    res = bot.respond("Zxyqwer plmokn 98765 randomgibberish?")
    assert res["source"] != "intent_model"


# ----------------------------------------------------
# 7. Flask API Tests
# ----------------------------------------------------
def test_api_chat_endpoint_contract(client):
    payload = {"message": "What is machine learning?", "conversation_id": "test_conv_1"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert "success" in data
    assert "answer" in data
    assert "source" in data
    assert data["success"] is True
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_api_chat_calculator(client):
    payload = {"message": "sqrt(144)"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert data["source"] == "calculator"
    assert "12" in data["answer"]


def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "gemini_available" in data
    assert data["has_intent_model"] is True
