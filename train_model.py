import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from nlp_utils import preprocess_text


def load_intents(data_path: str = "data/intents.json") -> dict:
    """Load intents dictionary from JSON file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Intent dataset not found at: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_training_data(intents_data: dict):
    """Extract and preprocess pattern sentences and their corresponding tags."""
    sentences = []
    labels = []
    for intent in intents_data.get("intents", []):
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            cleaned = preprocess_text(pattern)
            if cleaned:
                sentences.append(cleaned)
                labels.append(tag)
    return sentences, labels


def train_and_evaluate(
    data_path: str = "data/intents.json",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
):
    """Train TF-IDF Vectorizer and Logistic Regression model with evaluation."""
    print("=" * 60)
    print("NEXA-AI: TRAINING INTENT CLASSIFICATION MODEL")
    print("=" * 60)

    # 1. Load data
    intents_data = load_intents(data_path)
    sentences, labels = prepare_training_data(intents_data)
    print(f"Loaded {len(sentences)} training samples across {len(set(labels))} unique intent classes.")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        sentences,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    print(f"Training set: {len(X_train)} samples | Test evaluation set: {len(X_test)} samples.")

    # 3. TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=2500
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features.")

    # 4. Logistic Regression Training
    classifier = LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="lbfgs"
    )
    classifier.fit(X_train_tfidf, y_train)

    # 5. Model Evaluation
    y_pred = classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "-" * 40)
    print(f"EVALUATION ACCURACY: {acc * 100:.2f}%")
    print("-" * 40)
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Fit final model on full dataset for maximum deployment accuracy
    full_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=2500
    )
    X_full_tfidf = full_vectorizer.fit_transform(sentences)
    final_classifier = LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="lbfgs"
    )
    final_classifier.fit(X_full_tfidf, labels)

    # 7. Persist models
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "chatbot_model.pkl")
    vectorizer_file = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    with open(model_file, "wb") as f:
        pickle.dump(final_classifier, f)
    with open(vectorizer_file, "wb") as f:
        pickle.dump(full_vectorizer, f)

    print(f"\nSuccessfully saved model to: {model_file}")
    print(f"Successfully saved vectorizer to: {vectorizer_file}")
    print("=" * 60)
    return final_classifier, full_vectorizer, acc


if __name__ == "__main__":
    train_and_evaluate()
