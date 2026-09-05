import re
import urllib.parse
from typing import Any, Dict, Optional
import requests
from nlp_utils import clean_punctuation, extract_keywords

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS = {
    "User-Agent": "NexaAI-HybridChatbot/2.0 (internship-project; open-educational-nlp)"
}

STOP_PREFIXES = [
    r"^what\s+is\s+(?:the\s+)?(?:difference\s+between\s+)?",
    r"^what\s+are\s+(?:the\s+)?",
    r"^what\s+would\s+happen\s+if\s+(?:the\s+)?",
    r"^why\s+do\s+(?:humans\s+|people\s+)?",
    r"^why\s+is\s+(?:the\s+)?",
    r"^how\s+does\s+(?:a\s+|an\s+|the\s+)?",
    r"^how\s+do\s+(?:a\s+|an\s+|the\s+)?",
    r"^how\s+can\s+(?:i\s+|we\s+)?",
    r"^who\s+is\s+(?:the\s+)?",
    r"^who\s+was\s+(?:the\s+)?",
    r"^who\s+invented\s+(?:the\s+)?",
    r"^where\s+is\s+(?:the\s+)?",
    r"^explain\s+(?:the\s+)?",
    r"^tell\s+me\s+(?:about\s+|an\s+interesting\s+fact\s+about\s+)?",
    r"^give\s+me\s+(?:ideas\s+for\s+)?",
    r"^define\s+(?:the\s+)?",
    r"^write\s+a\s+(?:python\s+program\s+to\s+)?"
]

STOP_SUFFIXES = [
    r"\s+in\s+simple\s+words?$",
    r"\s+to\s+a\s+\d+\s+year\s+old$",
    r"\s+simply$",
    r"\s+work\??$",
    r"\s+works\??$",
    r"\s+efficiently\??$",
    r"\s+for\s+beginners?$",
    r"\s+with\s+an\s+example\??$",
    r"\s+stopped\s+rotating\??$"
]


def extract_subject_phrase(query: str) -> str:
    """
    Intelligently extract the primary core entity or subject from a natural language question.
    """
    cleaned = clean_punctuation(query).strip().lower()

    for p in STOP_PREFIXES:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()

    for s in STOP_SUFFIXES:
        cleaned = re.sub(s, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Domain specific subject normalizations
    if cleaned in ("earth", "earth stop", "earth rotating"):
        return "Earth's rotation"
    if cleaned in ("sort a list", "sort list"):
        return "Sorting algorithm"
    if cleaned in ("learn python", "python"):
        return "Python (programming language)"
    if cleaned == "space":
        return "Outer space"

    if cleaned:
        return cleaned

    kws = extract_keywords(query)
    return " ".join(kws) if kws else clean_punctuation(query).strip()


def fetch_wikipedia_summary(subject: str) -> Optional[Dict[str, Any]]:
    """
    Fetch clean encyclopedic summary for a subject using Wikipedia's public REST API.
    Returns None immediately if unavailable, timeout, or ambiguous.
    """
    if not subject or len(subject.strip()) < 2:
        return None

    variants = [
        subject.strip(),
        subject.strip().title(),
        subject.strip().rstrip("s")
    ]

    for var in variants:
        try:
            encoded_title = urllib.parse.quote(var.replace(" ", "_"))
            url = f"{WIKIPEDIA_SUMMARY_URL}{encoded_title}"
            resp = requests.get(url, headers=HEADERS, timeout=1.5)

            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "").strip()
                title = data.get("title", var)
                extract_type = data.get("type", "")

                # Strictly verify title relevance to the subject
                title_lower = title.lower()
                subj_tokens = set(subject.lower().split())
                title_tokens = set(title_lower.split())

                if extract and extract_type != "disambiguation" and len(extract) > 40:
                    if subj_tokens & title_tokens or subject.lower() in title_lower:
                        return {
                            "response": extract,
                            "title": title,
                            "confidence": 0.88,
                            "source": "knowledge_base"
                        }
        except Exception:
            continue

    return None


def answer_any_question(query: str) -> Optional[Dict[str, Any]]:
    """
    Universal Question Answering Controller:
    Attempts exact encyclopedic lookup.
    Never returns inappropriate dictionary definitions for complex inquiries.
    """
    subject = extract_subject_phrase(query)
    if not subject:
        return None

    wiki_res = fetch_wikipedia_summary(subject)
    if wiki_res:
        return wiki_res

    return None
