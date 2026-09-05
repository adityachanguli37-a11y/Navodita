# NexaAI: Intelligent Hybrid NLP & General AI Assistant

[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen.svg)](https://navodita-chatbot.onrender.com)
[![Render](https://img.shields.io/badge/deployed%20on-Render-46E3B7.svg)](https://navodita-chatbot.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

🌐 **Live Demo:** [https://navodita-chatbot.onrender.com](https://navodita-chatbot.onrender.com)

An enterprise-grade **Intelligent Hybrid Conversational AI Assistant** combining the precision of a safe AST mathematical calculator, conservative conversational intent classification, strict local knowledge retrieval, and the universal intelligence of **Google Gemini API** (`gemini-3.5-flash-lite`) with transparent graceful fallbacks.

---

## 🚀 Live Deployment

The application is deployed live on Render free tier:
- **Web App:** [https://navodita-chatbot.onrender.com](https://navodita-chatbot.onrender.com)
- **Health Check:** [https://navodita-chatbot.onrender.com/health](https://navodita-chatbot.onrender.com/health)

*(Note: On Render's free tier, the instance may spin down after 15 minutes of inactivity. Please allow ~30 seconds for the initial cold start).*

---

## 💡 System Architecture

NexaAI guarantees that **for every question, the final answer directly and accurately addresses that question**. The system rejects intent hijacking and weak local matches, routing general inquiries to Google Gemini.

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
                                                    [ Google Gemini AI ]
                                                  (Official google.genai SDK:
                                                   gemini-3.5-flash-lite)
                                                   Source: "gemini"
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
1. **Google Gemini Universal AI**: Uses the official `google.genai` SDK with `gemini-3.5-flash-lite` for universal knowledge, coding, science, and general question answering.
2. **Strict Intent Hijacking Prevention**: The ML intent classifier (TF-IDF + Logistic Regression) is strictly restricted to conversational markers (`hello`, `thanks`, `goodbye`) and explicit bot operations. It is never allowed to hijack general, informational, or technical questions.
3. **Strictly Gated Local Knowledge Base**: Local documents are returned only if there is high semantic similarity AND explicit topic relevance. If a question is not directly covered, it is never served a loose answer.
4. **Safe Mathematical Calculator**: Safely parses arithmetic expressions into an Abstract Syntax Tree (AST), rejecting `eval()` and arbitrary code completely. Questions *about* math (e.g. "Why does multiplication work?") go to Gemini.
5. **Multi-Turn Context & Topic Switching**: Resolves follow-up pronouns (`"What are its advantages?"`) while cleanly resetting context when the user switches topics.
6. **Confidence-Aware Truthful Fallback**: Transparently informs the user if a query cannot be answered locally and the cloud AI is offline, preventing hallucinated or misleading dictionary definitions.

---

## 📂 Project Structure

```
Navodita/
├── app.py                     # Flask web server and REST API endpoints (/chat, /health)
├── chatbot.py                 # HybridChatbot controller coordinating all layers
├── knowledge_engine.py        # Strict TF-IDF knowledge base retriever & context resolver
├── gemini_service.py          # Google Gemini service wrapper (google.genai SDK)
├── calculator.py              # Safe AST-based mathematical expression parser
├── nlp_utils.py               # Tokenization, lemmatization, stop-words, cleaning
├── universal_qa.py            # Encyclopedic question answering engine
├── train_model.py             # Script to train and evaluate ML intent classifier
├── render.yaml                # Render Infrastructure-as-Code deployment specification
├── Procfile                   # Process file for Gunicorn production deployment
├── requirements.txt           # Python dependencies (Flask, google-genai, gunicorn, etc.)
├── .env.example               # Template for environment configuration
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
    └── test_chatbot.py        # Comprehensive pytest automated unit & integration tests
```

---

## 🛠️ Local Installation & Setup

### 1. Prerequisites
- Python 3.10+ installed.
- Free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```

In `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 4. Train the Intent Classification Model
```bash
python train_model.py
```

### 5. Run Automated Tests
```bash
pytest -v
```

### 6. Start the Web Application
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
  "answer": "Quantum computing is a multidisciplinary field...",
  "response": "Quantum computing is a multidisciplinary field...",
  "source": "gemini",
  "confidence": 0.98,
  "conversation_id": "optional-uuid"
}
```

### `GET /health`
Returns system status, Gemini availability, model name, and knowledge base topic count:
```json
{
  "status": "healthy",
  "gemini_available": true,
  "model": "gemini-3.5-flash-lite",
  "kb_topics_loaded": 10
}
```
