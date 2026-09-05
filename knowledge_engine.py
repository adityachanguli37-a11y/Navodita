import json
import os
import re
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_utils import extract_keywords, preprocess_text

FOLLOWUP_PRONOUNS = {"it", "its", "it's", "this", "that", "these", "those", "they", "them", "their"}
FOLLOWUP_PHRASES = (
    "what about", "how about", "give me an example", "give an example",
    "tell me more", "explain more", "more details", "what are advantages",
    "what are the advantages", "what are its advantages", "what are benefits",
    "any example", "another example"
)
COMPARISON_WORDS = {"difference", "differ", "vs", "versus", "compare", "comparison", "between"}


class KnowledgeEngine:
    def __init__(self, kb_path: str = "data/knowledge_base.json"):
        self.kb_path = kb_path
        self.documents: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.load_and_index()

    def load_and_index(self):
        """Load knowledge base JSON and fit TF-IDF vectorizer over the corpus."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.doc_texts = []
        for doc in self.documents:
            title = doc.get("title", "")
            keywords = " ".join(doc.get("keywords", []))
            summary = doc.get("summary", "")
            content = doc.get("content", "")
            combined_text = f"{title} {keywords} {summary} {content}"
            self.doc_texts.append(preprocess_text(combined_text))

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def extract_recent_topic(self, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """
        Extract the main subject or topic entity from the most recent conversation turn.
        Prioritizes user turns so words in assistant explanations do not distort topic tracking.
        """
        if not history or not isinstance(history, list):
            return None

        # Look backward through user turns first
        for msg in reversed(history[-6:]):
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "").strip()
            if not content:
                continue

            c_lower = content.lower()

            # Priority 1: Check known multi-word topics or keywords
            for doc in self.documents:
                topic_words = doc.get("topic", "").replace("_", " ").lower()
                if topic_words in c_lower:
                    return topic_words
                for kw in doc.get("keywords", []):
                    if len(kw) >= 3 and re.search(r"\b" + re.escape(kw.lower()) + r"\b", c_lower):
                        return kw.lower()

            # Priority 2: Extract non-stopword entity tokens
            kws = extract_keywords(content)
            if kws:
                return kws[0]

        return None

    def resolve_context(self, query: str, history: Optional[List[Dict[str, str]]]) -> str:
        """
        Resolve conversation context for follow-up queries.
        Only augments if the query is genuinely an anaphoric follow-up (e.g. 'What are its advantages?').
        NEVER augments if the user has switched topics or introduced a new subject.
        """
        if not history or not isinstance(history, list):
            return query

        q_clean = query.strip().lower()
        q_tokens = set(re.findall(r"\b[a-zA-Z]+\b", q_clean))

        has_pronoun = bool(q_tokens & FOLLOWUP_PRONOUNS)
        is_followup_phrase = any(q_clean.startswith(p) or f" {p} " in f" {q_clean} " for p in FOLLOWUP_PHRASES)

        # Only augment if it's a pronoun or follow-up phrase without introducing a separate new subject
        if (has_pronoun and len(q_tokens) <= 7) or (is_followup_phrase and len(q_tokens) <= 6):
            topic = self.extract_recent_topic(history)
            if topic and topic not in q_clean:
                return f"{query} {topic}"

        return query

    def query(
        self,
        raw_query: str,
        history: Optional[List[Dict[str, str]]] = None,
        min_threshold: float = 0.35
    ) -> Optional[Dict[str, Any]]:
        """
        Perform strict extractive knowledge retrieval.
        Returns a document only if there is both high semantic similarity AND explicit topic relevance.
        Rejects weak matches to allow the general AI engine (Gemini / Ollama) to answer.
        """
        if not raw_query or not raw_query.strip():
            return None

        resolved_query = self.resolve_context(raw_query, history)
        processed_query = preprocess_text(resolved_query)

        if not processed_query.strip() or self.vectorizer is None or self.tfidf_matrix is None:
            return None

        query_vec = self.vectorizer.transform([processed_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        query_lower = resolved_query.lower()
        query_has_comparison = bool(set(re.findall(r"\b[a-zA-Z]+\b", query_lower)) & COMPARISON_WORDS)
        query_has_simple = any(w in query_lower for w in ("simple", "simply", "layman", "kid", "5 year"))

        adjusted_scores = np.copy(sim_scores)

        for i, doc in enumerate(self.documents):
            topic = doc.get("topic", "")
            keywords = [k.lower() for k in doc.get("keywords", [])]

            # 1. Comparison handling:
            if topic == "ai_vs_ml":
                if query_has_comparison and ("ai" in query_lower or "artificial intelligence" in query_lower) and ("ml" in query_lower or "machine learning" in query_lower):
                    adjusted_scores[i] += 0.35
                else:
                    adjusted_scores[i] -= 0.50

            # 2. Simple neural network handling: strictly require neural/network in query
            if topic == "neural_networks_simple":
                if query_has_simple and ("neural" in query_lower or "network" in query_lower):
                    adjusted_scores[i] += 0.35
                else:
                    adjusted_scores[i] -= 0.50

            # 3. Topic specific bonuses
            if topic == "python_advantages" and ("advantage" in query_lower or "benefit" in query_lower or "feature" in query_lower) and "python" in query_lower:
                adjusted_scores[i] += 0.35
            elif topic == "telephone_invention" and ("telephone" in query_lower or "phone" in query_lower) and ("invent" in query_lower or "who" in query_lower):
                adjusted_scores[i] += 0.35
            elif topic == "capital_of_france" and ("france" in query_lower or "paris" in query_lower) and ("capital" in query_lower or "city" in query_lower):
                adjusted_scores[i] += 0.35
            elif topic == "why_sky_is_blue" and ("sky" in query_lower and "blue" in query_lower):
                adjusted_scores[i] += 0.35
            elif topic == "how_internet_works" and "internet" in query_lower and ("how" in query_lower or "work" in query_lower):
                adjusted_scores[i] += 0.35
            elif topic == "recursion" and "recursion" in query_lower:
                adjusted_scores[i] += 0.35
            elif topic == "jokes" and ("joke" in query_lower or "funny" in query_lower):
                adjusted_scores[i] += 0.35
            elif topic == "blockchain" and "blockchain" in query_lower:
                adjusted_scores[i] += 0.35
            elif topic == "quantum_computing" and "quantum computing" in query_lower and "entanglement" not in query_lower:
                adjusted_scores[i] += 0.35

            # 4. Exact keyword phrase matching bonus:
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                    adjusted_scores[i] += 0.15 + (0.05 * len(kw.split()))
                    break

        best_idx = int(np.argmax(adjusted_scores))
        best_score = float(adjusted_scores[best_idx])
        best_doc = self.documents[best_idx]

        # STRICT RELEVANCE VALIDATION:
        # The document must have direct, unambiguous topical relevance to the query entity.
        topic = best_doc.get("topic", "")
        keywords = best_doc.get("keywords", [])

        is_topically_relevant = False

        # Check explicit topic entity matches
        if topic == "artificial_intelligence" and ("artificial intelligence" in query_lower or query_lower.strip() in ("what is ai", "ai", "what is ai?")) and not query_has_comparison:
            is_topically_relevant = True
        elif topic == "machine_learning" and "machine learning" in query_lower and not query_has_comparison:
            is_topically_relevant = True
        elif topic == "deep_learning" and "deep learning" in query_lower:
            is_topically_relevant = True
        elif topic == "natural_language_processing" and ("natural language processing" in query_lower or "nlp" in query_lower):
            is_topically_relevant = True
        elif topic == "ai_vs_ml" and query_has_comparison and ("ai" in query_lower or "artificial intelligence" in query_lower) and ("ml" in query_lower or "machine learning" in query_lower):
            is_topically_relevant = True
        elif topic == "neural_networks_simple" and ("neural" in query_lower and "network" in query_lower):
            is_topically_relevant = True
        elif topic == "python_programming" and "python" in query_lower and not ("advantage" in query_lower or "benefit" in query_lower):
            is_topically_relevant = True
        elif topic == "python_advantages" and "python" in query_lower and ("advantage" in query_lower or "benefit" in query_lower or "feature" in query_lower):
            is_topically_relevant = True
        elif topic == "recursion" and "recursion" in query_lower:
            is_topically_relevant = True
        elif topic == "database_systems" and ("database" in query_lower or "dbms" in query_lower) and "sql" not in query_lower and "mongo" not in query_lower:
            is_topically_relevant = True
        elif topic == "sql" and "sql" in query_lower:
            is_topically_relevant = True
        elif topic == "mongodb" and ("mongodb" in query_lower or "mongo" in query_lower):
            is_topically_relevant = True
        elif topic == "cybersecurity" and ("cybersecurity" in query_lower or "infosec" in query_lower or "cia triad" in query_lower):
            is_topically_relevant = True
        elif topic == "cloud_computing" and "cloud computing" in query_lower:
            is_topically_relevant = True
        elif topic == "computer_networks" and ("computer network" in query_lower or "osi model" in query_lower or "tcp/ip" in query_lower):
            is_topically_relevant = True
        elif topic == "how_internet_works" and "internet" in query_lower and ("how" in query_lower or "work" in query_lower):
            is_topically_relevant = True
        elif topic == "operating_systems" and ("operating system" in query_lower or " os " in f" {query_lower} "):
            is_topically_relevant = True
        elif topic == "data_structures" and "data structure" in query_lower:
            is_topically_relevant = True
        elif topic == "algorithms" and "algorithm" in query_lower and "sorting" not in query_lower:
            is_topically_relevant = True
        elif topic == "telephone_invention" and ("telephone" in query_lower or "phone" in query_lower) and ("invent" in query_lower or "who" in query_lower):
            is_topically_relevant = True
        elif topic == "capital_of_france" and ("france" in query_lower or "paris" in query_lower) and ("capital" in query_lower or "city" in query_lower or "where" in query_lower):
            is_topically_relevant = True
        elif topic == "why_sky_is_blue" and ("sky" in query_lower and "blue" in query_lower):
            is_topically_relevant = True
        elif topic == "jokes" and ("joke" in query_lower or "funny" in query_lower):
            is_topically_relevant = True
        elif topic == "blockchain" and "blockchain" in query_lower:
            is_topically_relevant = True
        elif topic == "quantum_computing" and "quantum computing" in query_lower and "entanglement" not in query_lower:
            is_topically_relevant = True
        else:
            for kw in keywords:
                if len(kw) >= 3 and re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                    is_topically_relevant = True
                    break

        if not is_topically_relevant:
            return None

        if best_score < min_threshold:
            return None

        calibrated_conf = min(0.98, max(0.75, round(0.65 + min(0.30, best_score * 0.35), 2)))
        response_text = self._format_response(raw_query, best_doc)

        return {
            "response": response_text,
            "confidence": calibrated_conf,
            "raw_score": round(best_score, 4),
            "topic": best_doc.get("topic", "general_knowledge"),
            "title": best_doc.get("title", "")
        }

    def _format_response(self, query: str, doc: Dict[str, Any]) -> str:
        """Format an extractive or comprehensive answer based on the nature of the question."""
        content = doc.get("content", "")
        summary = doc.get("summary", "")
        q_lower = query.lower()

        if "simple" in q_lower or "brief" in q_lower or "summary" in q_lower:
            if doc.get("topic") == "neural_networks_simple" and ("neural" in q_lower or "network" in q_lower):
                return content
            if summary:
                return f"{summary} {content.split('.')[0]}."

        return content
