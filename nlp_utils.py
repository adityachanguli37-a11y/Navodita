import re
import unicodedata

# Preserved question & comparison tokens that should NOT be stripped as generic stopwords
PRESERVED_QUERY_WORDS = {
    "what", "why", "how", "who", "where", "when", "which",
    "difference", "vs", "versus", "between", "explain", "advantages",
    "simple", "definition", "work", "works", "invented", "capital",
    "blue", "sky", "joke", "internet"
}

STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how's", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what's", "when's", "where's", "which", "while", "who's",
    "whom", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Effective stopwords are standard stopwords minus preserved query tokens
FILTER_STOPWORDS = {w for w in STANDARD_STOPWORDS if w not in PRESERVED_QUERY_WORDS}

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    _lemmatizer = WordNetLemmatizer()
    HAS_NLTK = True
except Exception:
    _lemmatizer = None
    HAS_NLTK = False


def normalize_text(text: str) -> str:
    """Normalize unicode characters, lowercase, and clean excess whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_punctuation(text: str) -> str:
    """Remove special punctuation while preserving alphanumerics and essential spaces."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text using NLTK word_tokenize with regex fallback."""
    if not text:
        return []
    cleaned = clean_punctuation(text)
    if HAS_NLTK:
        try:
            return [t for t in word_tokenize(cleaned) if t]
        except Exception:
            pass
    return [t for t in cleaned.split() if t]


def lemmatize_token(token: str) -> str:
    """Lemmatize token with WordNetLemmatizer or basic rule fallback."""
    if not token:
        return ""
    if HAS_NLTK and _lemmatizer is not None:
        try:
            # Try verb first then noun
            lemma = _lemmatizer.lemmatize(token, pos="v")
            if lemma == token:
                lemma = _lemmatizer.lemmatize(token, pos="n")
            return lemma
        except Exception:
            pass
    # Basic morphological fallbacks
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """Full preprocessing pipeline: normalize, tokenize, lemmatize, optional stopword filter."""
    norm = normalize_text(text)
    tokens = tokenize(norm)
    processed = []
    for tok in tokens:
        if remove_stopwords and tok in FILTER_STOPWORDS:
            continue
        processed.append(lemmatize_token(tok))
    return " ".join(processed)


def extract_keywords(text: str) -> list[str]:
    """Extract key non-stopword normalized tokens representing main topics/entities."""
    norm = normalize_text(text)
    tokens = tokenize(norm)
    keywords = [tok for tok in tokens if tok not in FILTER_STOPWORDS and len(tok) > 1]
    return keywords
