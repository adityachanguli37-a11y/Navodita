# NexaAI: Intelligent Hybrid NLP & General AI Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![OpenAI](https://img.shields.io/badge/API-OpenAI%20Responses-green.svg)](https://platform.openai.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen.svg)](tests/)

An enterprise-grade **Intelligent Hybrid Conversational AI Assistant** combining the precision of a safe AST mathematical calculator, conservative conversational intent classification, strict local knowledge retrieval, and the universal intelligence of **OpenAI (Responses API & Chat Completions)** with transparent graceful fallbacks.

---

## 💡 System Architecture

NexaAI guarantees that **for every question, the final answer directly and accurately addresses that question**. The system rejects intent hijacking and weak local matches, routing general inquiries to OpenAI.

```
                           [ USER QUESTION ]
                                  │
                                  ▼
                        [ Input Validation ]
                      (Rejects empty / whitespace)
                                  │
                                  ▼
                   [ Context & Topic Resolution ]
               (Resolves 'it'/'its' or cleanly switches)
                                  │
                                  ▼
                    [ Question Type Detection ]
                                  │
      ┌───────────────────────────┼──────────────────────────┐
      ▼                           ▼                          ▼
[ Exact Math AST ]     [ Conversational Intent ]    [ General Inquiry ]
(25 * 8, sqrt(144))    (hello, thanks, bye, hours)  (what, why, how, code, etc.)
Source: "calculator"      Source: "intent_model"             │
                                                             ▼
                                                    [ OpenAI General AI ]
                                                   (Official Python SDK:
                                                    Responses API & Chat API)
                                                    Source: "openai"
                                                             │
                                                  (If Offline / No Key)
                                                             │
                                                             ▼
                                                    [ Strict Local KB ]
                                                    (Only exact topic matches)
                                                    Source: "knowledge_base"
                                                             │
                                                  (If Topic Unmatched)
                                                             │
                                                             ▼
                                                    [ Truthful Fallback ]
                                                    Source: "fallback"
```

### Core Architecture Pillars:
1. **OpenAI Universal General AI**: Uses the official OpenAI Python SDK supporting both the modern Responses API (`client.responses.create`) and Chat Completions (`client.chat.completions.create`). Configurable via `OPENAI_MODEL` with zero hardcoded keys.
2. **Strict Intent Hijacking Prevention**: The ML intent classifier (TF-IDF + Logistic Regression) is strictly restricted to conversational markers (`hello`, `thanks`, `goodbye`) and explicit bot operations. It is never allowed to hijack general, informational, or technical questions.
3. **Strictly Gated Local Knowledge Base**: Local documents are returned only if there is high semantic similarity AND explicit topic relevance. If a question is not directly covered, it is never served a loose answer.
4. **Safe Mathematical Calculator**: Safely parses arithmetic expressions into an Abstract Syntax Tree (AST), rejecting `eval()` and arbitrary code completely. Questions *about* math (e.g. "Why does multiplication work?") go to OpenAI.
5. **Multi-Turn Context & Topic Switching**: Resolves follow-up pronouns (`"What are its advantages?"`) while cleanly resetting context when the user switches topics (e.g. from Machine Learning to Black Holes).
6. **Confidence-Aware Truthful Fallback**: Transparently informs the user if a query cannot be answered locally and the cloud AI is offline, preventing hallucinated or misleading dictionary definitions.

---

## 🚀 Features

- **Accurate Universal Question Answering**: Understands the user's actual question and answers it directly.
- **OpenAI Responses API Integration**: Seamless integration with OpenAI models (default: `gpt-4o-mini`).
- **Safe Calculator**: Evaluates arithmetic expressions (`+`, `-`, `*`, `/`, `%`, `**`, `sqrt`, `()`) without security vulnerabilities.
- **Transparent Source Attribution**: Every response clearly indicates its origin (`Source: OpenAI Intelligence`, `Source: Knowledge Base`, `Source: Intent Model`, `Source: Calculator`).
- **Multi-Turn Conversation**: Context-aware follow-ups with bounded session history.
- **Modern Responsive Web UI**: Glassmorphism dark-theme interface with markdown rendering and code block styling.

---

## 📂 Project Structure

```
Navodita/
├── app.py                     # Flask web server and REST API endpoints (/chat, /health)
├── chatbot.py                 # HybridChatbot controller coordinating all layers
├── knowledge_engine.py        # Strict TF-IDF knowledge base retriever & context resolver
├── openai_service.py          # OpenAI Responses API & Chat Completions service wrapper
├── calculator.py              # Safe AST-based mathematical expression parser
├── nlp_utils.py               # Tokenization, lemmatization, stop-words, cleaning
├── universal_qa.py            # Encyclopedic question answering engine
├── train_model.py             # Script to train and evaluate ML intent classifier
├── verify_questions.py        # Verification script testing all required questions
├── data/
│   ├── intents.json           # Labeled conversational patterns
│   └── knowledge_base.json    # Local technical and scientific knowledge documents
├── models/
│   ├── chatbot_model.pkl      # Serialized Logistic Regression intent classifier
│   └── tfidf_vectorizer.pkl   # Serialized TF-IDF feature extractor
├── templates/
│   └── index.html             # HTML5 conversational user interface
├── static/
│   ├── css/style.css          # Glassmorphism dark-theme styling
│   └── js/chat.js             # Client state management, AJAX chat, history tracker
└── tests/
    └── test_chatbot.py        # 27 comprehensive pytest automated unit & integration tests
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install flask scikit-learn nltk pytest requests openai
```

### 3. Environment Variables Configuration
To enable the OpenAI general AI engine, set your API key in the environment:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-actual-api-key"
$env:OPENAI_MODEL = "gpt-4o-mini"   # Optional, defaults to gpt-4o-mini
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-actual-api-key
set OPENAI_MODEL=gpt-4o-mini
```

**Linux / macOS:**
```bash
export OPENAI_API_KEY="your-actual-api-key"
export OPENAI_MODEL="gpt-4o-mini"
```

> **Note**: If `OPENAI_API_KEY` is not provided, the chatbot operates gracefully in offline mode using the safe AST calculator and strictly verified local knowledge.

### 4. Train the Intent Classification Model
```bash
python train_model.py
```

### 5. Run Automated Tests
```bash
pytest -v
```

### 6. Run Full Question Verification
```bash
python verify_questions.py
```

### 7. Start the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.

---

## 🔌 API Specification

### `POST /chat`
Sends a message and optional conversation history.

**Request Body:**
```json
{
  "message": "Explain quantum computing",
  "conversation_id": "optional-uuid",
  "history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hello! How can I help you?" }
  ]
}
```

**Response Body:**
```json
{
  "success": true,
  "answer": "Quantum computing is a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than classical computers...",
  "response": "Quantum computing is a multidisciplinary field...",
  "source": "openai",
  "confidence": 0.98,
  "conversation_id": "optional-uuid"
}
```

### `GET /health`
Returns system status, OpenAI availability, model name, and knowledge base topic count.

---

## 🧪 Evaluation & Test Coverage

The test suite in [`tests/test_chatbot.py`](file:///c:/Users/adity/OneDrive/Desktop/Projects/Navodita/tests/test_chatbot.py) includes 27 comprehensive tests:
- **Math AST Calculator**: Arithmetic operations, powers, parentheses, square roots, division-by-zero handling, code injection rejection.
- **Intent Hijacking Rejection**: Proves that arbitrary informational and general questions are NEVER answered with intent canned responses, even when the ML classifier produces 99% confidence for an unrelated intent.
- **Conversational Intents**: Accurate detection of greetings, goodbyes, thanks, business hours, and contact inquiries.
- **OpenAI Responses API & Chat Completions**: Unit and mock tests verifying correct API call parameters and system prompt delivery.
- **Knowledge Base Gating**: Strict topic-entity matching and rejection of unrelated queries.
- **Contextual Follow-ups**: Anaphoric resolution ("What are its advantages?") across conversation turns.
- **Topic Switching**: Verifies that when a user switches topics, previous context does not pollute the new question.
- **Flask API Contract**: Complete verification of `/chat` and `/health` response formats and error handling.
