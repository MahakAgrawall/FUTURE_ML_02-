"""
=============================================================
  Support Ticket Classification & Priority Prediction System
  Task 2 — Future Interns ML Project
=============================================================
  Features:
    - Synthetic dataset generation (no external download needed)
    - Text cleaning & tokenization (NLTK)
    - TF-IDF feature extraction
    - Multi-class category classification
    - Priority level prediction (High / Medium / Low)
    - Full evaluation: accuracy, precision, recall, F1
    - Confusion matrix & class-wise report
    - Single-ticket demo inference
=============================================================
"""

# ──────────────────────────────────────────────
# 1.  IMPORTS
# ──────────────────────────────────────────────
import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # headless backend – safe for scripts
warnings.filterwarnings("ignore")

# NLTK
import nltk
for pkg in ("stopwords", "punkt", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
# 2.  SYNTHETIC DATASET
# ──────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

TEMPLATES = {
    "Billing": [
        "I was charged twice for my subscription this month.",
        "My invoice shows an incorrect amount. Please correct it.",
        "I need a refund for the duplicate payment made yesterday.",
        "Why was my credit card charged without any notification?",
        "I cancelled my plan but I am still being billed.",
        "Promo code was not applied during checkout. Please adjust.",
        "I cannot download my invoice PDF from the billing portal.",
        "Payment failed but money was deducted from my account.",
        "I want to upgrade my plan and need to know the price difference.",
        "Tax calculation on my invoice appears to be wrong.",
    ],
    "Technical": [
        "The application crashes every time I try to upload a file.",
        "Login page shows 500 internal server error since this morning.",
        "API endpoint is returning 404 not found for a valid request.",
        "Dashboard widgets are not loading after the latest update.",
        "I cannot connect to the database; getting timeout errors.",
        "Two-factor authentication is broken and not sending OTP.",
        "Export to CSV feature is producing empty files.",
        "Mobile app freezes on the home screen after logging in.",
        "Email notifications are not being delivered to my inbox.",
        "Integration with Slack stopped working after your recent release.",
    ],
    "Account": [
        "I forgot my password and the reset email is not arriving.",
        "Please help me delete my account and all associated data.",
        "I need to change the email address linked to my account.",
        "My account was locked after too many failed login attempts.",
        "I want to add a team member but the invite button is missing.",
        "How do I transfer ownership of the account to another user?",
        "Profile picture upload is failing with an unsupported format error.",
        "I need to update my billing address in the account settings.",
        "Two accounts were accidentally merged. Please separate them.",
        "I signed up with Google but can no longer access Google SSO.",
    ],
    "General": [
        "Can you explain how the reporting feature works?",
        "Where can I find the documentation for the REST API?",
        "What are the supported file formats for data import?",
        "I would like to request a live demo of your enterprise plan.",
        "How long does onboarding typically take for a team of 50?",
        "Do you offer a non-profit discount for registered charities?",
        "Is there a roadmap for adding multi-language support?",
        "What is the maximum file size allowed for uploads?",
        "Can I use your service if I am based outside the United States?",
        "How do I export all my data before cancelling my subscription?",
    ],
}

PRIORITY_MAP = {
    "Billing":   {"High": 0.3, "Medium": 0.5, "Low": 0.2},
    "Technical": {"High": 0.5, "Medium": 0.35, "Low": 0.15},
    "Account":   {"High": 0.25, "Medium": 0.45, "Low": 0.3},
    "General":   {"High": 0.1, "Medium": 0.3, "Low": 0.6},
}

def _pick_priority(category: str) -> str:
    dist = PRIORITY_MAP[category]
    return random.choices(list(dist.keys()), weights=list(dist.values()))[0]

def generate_dataset(n: int = 800) -> pd.DataFrame:
    """Generate a synthetic support-ticket dataset of size n."""
    categories = list(TEMPLATES.keys())
    rows = []
    for _ in range(n):
        cat = random.choice(categories)
        base = random.choice(TEMPLATES[cat])
        # Light augmentation: randomly prepend a subject-line style phrase
        prefixes = [
            "", "", "",                          # keep original most often
            "URGENT: ", "Re: ", "FWD: ",
            "Hello support team, ", "Hi, ",
        ]
        text = random.choice(prefixes) + base
        rows.append({"text": text, "category": cat,
                     "priority": _pick_priority(cat)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3.  TEXT PREPROCESSING
# ──────────────────────────────────────────────
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Pipeline:
      lowercase → strip HTML → remove special chars →
      tokenise → remove stopwords → lemmatise → rejoin
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens
              if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# 4.  MODEL BUILDING HELPERS
# ──────────────────────────────────────────────
def build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])

MODELS = {
    "Logistic Regression": build_pipeline(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    ),
    "Naive Bayes": build_pipeline(
        MultinomialNB(alpha=0.1)
    ),
    "Random Forest": build_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    ),
}


# ──────────────────────────────────────────────
# 5.  EVALUATION & PLOTTING
# ──────────────────────────────────────────────
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test,
                   label: str = "Category") -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{'─'*55}")
    print(f"  {name}  —  {label} Classifier")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return {"name": name, "accuracy": acc, "pipeline": pipeline,
            "report": report, "y_test": y_test, "y_pred": y_pred}


def plot_confusion_matrix(result: dict, label: str, filename: str):
    cm = confusion_matrix(result["y_test"], result["y_pred"],
                          labels=result["pipeline"].classes_)
    disp = ConfusionMatrixDisplay(cm,
                                  display_labels=result["pipeline"].classes_)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f'{result["name"]} — {label} Confusion Matrix', pad=12,
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [saved] {filename}")


def plot_model_comparison(results: list, label: str, filename: str):
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, accs, color=["#3B82F6", "#10B981", "#F59E0B"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(f"Model Comparison — {label}", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [saved] {filename}")


# ──────────────────────────────────────────────
# 6.  SINGLE-TICKET INFERENCE
# ──────────────────────────────────────────────
def predict_ticket(text: str,
                   cat_pipeline,
                   pri_pipeline) -> dict:
    cleaned = clean_text(text)
    category = cat_pipeline.predict([cleaned])[0]
    cat_proba = dict(zip(cat_pipeline.classes_,
                         cat_pipeline.predict_proba([cleaned])[0]))
    priority = pri_pipeline.predict([cleaned])[0]
    pri_proba = dict(zip(pri_pipeline.classes_,
                         pri_pipeline.predict_proba([cleaned])[0]))
    return {
        "original_text": text,
        "cleaned_text":  cleaned,
        "category":      category,
        "category_confidence": f"{max(cat_proba.values()):.1%}",
        "priority":      priority,
        "priority_confidence": f"{max(pri_proba.values()):.1%}",
    }


# ──────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Support Ticket Classifier — Task 2")
    print("=" * 55)

    # 7.1  Generate & inspect data
    print("\n[1/6] Generating synthetic dataset …")
    df = generate_dataset(n=1000)
    print(df.head(8).to_string(index=False))
    print(f"\n  Shape  : {df.shape}")
    print(f"  Categories : {df['category'].value_counts().to_dict()}")
    print(f"  Priorities : {df['priority'].value_counts().to_dict()}")

    # 7.2  Clean text
    print("\n[2/6] Cleaning and preprocessing text …")
    df["clean_text"] = df["text"].apply(clean_text)

    # 7.3  Encode labels
    le_cat = LabelEncoder()
    le_pri = LabelEncoder()
    df["cat_label"] = le_cat.fit_transform(df["category"])
    df["pri_label"] = le_pri.fit_transform(df["priority"])

    # 7.4  Train / test split
    X = df["clean_text"]
    y_cat = df["category"]
    y_pri = df["priority"]

    X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
        X, y_cat, y_pri, test_size=0.2, random_state=42, stratify=y_cat
    )
    print(f"\n  Train size : {len(X_train)}   Test size : {len(X_test)}")

    # 7.5  Train & evaluate — CATEGORY
    print("\n[3/6] Training category classifiers …")
    cat_results = []
    for name, pipe in MODELS.items():
        import copy
        r = evaluate_model(name, copy.deepcopy(pipe),
                           X_train, X_test, yc_train, yc_test,
                           label="Category")
        cat_results.append(r)

    best_cat = max(cat_results, key=lambda x: x["accuracy"])
    print(f"\n  ✅ Best category model : {best_cat['name']}  "
          f"(acc = {best_cat['accuracy']:.4f})")

    # 7.6  Train & evaluate — PRIORITY
    print("\n[4/6] Training priority classifiers …")
    pri_results = []
    for name, pipe in MODELS.items():
        import copy
        r = evaluate_model(name, copy.deepcopy(pipe),
                           X_train, X_test, yp_train, yp_test,
                           label="Priority")
        pri_results.append(r)

    best_pri = max(pri_results, key=lambda x: x["accuracy"])
    print(f"\n  ✅ Best priority model : {best_pri['name']}  "
          f"(acc = {best_pri['accuracy']:.4f})")

    # 7.7  Plots
    print("\n[5/6] Generating evaluation plots …")
    plot_confusion_matrix(best_cat, "Category",
                          "/mnt/user-data/outputs/confusion_category.png")
    plot_confusion_matrix(best_pri, "Priority",
                          "/mnt/user-data/outputs/confusion_priority.png")
    plot_model_comparison(cat_results, "Category",
                          "/mnt/user-data/outputs/model_comparison_category.png")
    plot_model_comparison(pri_results, "Priority",
                          "/mnt/user-data/outputs/model_comparison_priority.png")

    # 7.8  Demo inference
    print("\n[6/6] Demo — predicting new tickets …\n")
    sample_tickets = [
        "I have been charged twice this month and need an immediate refund!",
        "The login page shows a 500 error and I cannot access my account.",
        "How do I reset my password? The email link is not working.",
        "Can you explain what features are included in the Pro plan?",
        "URGENT: Our entire team is locked out after your latest update.",
    ]

    for ticket in sample_tickets:
        result = predict_ticket(ticket,
                                best_cat["pipeline"],
                                best_pri["pipeline"])
        print(f"  Ticket   : {result['original_text']}")
        print(f"  Category : {result['category']}  "
              f"(confidence {result['category_confidence']})")
        print(f"  Priority : {result['priority']}  "
              f"(confidence {result['priority_confidence']})")
        print()

    print("=" * 55)
    print("  All done! Files saved to /mnt/user-data/outputs/")
    print("=" * 55)


if __name__ == "__main__":
    main()
