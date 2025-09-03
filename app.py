import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import numpy as np

st.set_page_config(page_title="Spam Detector", page_icon="📧")

# ----------------------------
# Train once (cached)
# ----------------------------
@st.cache_resource
def train_model():
    # Load + minimal prep (no NLTK needed)
    df = pd.read_csv("spam.csv", encoding="latin1")
    df = df.rename(columns={"v1": "target", "v2": "text"})[["target", "text"]]
    df["target"] = df["target"].map({"ham": 0, "spam": 1}).astype(int)

    # FeatureUnion: word n-grams + char n-grams (great for URLs/numbers)
    feats = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),              # unigrams + bigrams ("click here")
            token_pattern=r"(?u)\b\w+\b",     # keep digits & single-char tokens
            min_df=2, max_df=0.95,
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char",                  # keeps ., /, :, etc. for URLs
            ngram_range=(3, 5),               # character 3-5 grams
            min_df=2,
            sublinear_tf=True
        )),
    ])

    clf = LogisticRegression(max_iter=2000, C=2.0, n_jobs=None)
    pipe = Pipeline([("feats", feats), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )
    pipe.fit(X_train, y_train)

    # Find a data-driven threshold (maximize F1 on validation)
    proba = pipe.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_tau = float(thr[np.argmax(f1)]) if thr.size else 0.5

    return pipe, best_tau

pipe, default_tau = train_model()

# ----------------------------
# UI
# ----------------------------
st.title("📧 Spam Message Detector")
msg = st.text_area("Enter a message:", height=120, value="wow, you got 1500 dollars just click here www.xyz.com")

col1, col2 = st.columns([2,1])
with col1:
    tau = st.slider("Spam decision threshold (lower → more aggressive)", 0.05, 0.95, float(default_tau), 0.01)
with col2:
    st.metric("Default best F1 τ", f"{default_tau:.2f}")

if st.button("Predict"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        p = pipe.predict_proba([msg])[0, 1]
        pred = int(p >= tau)
        if pred == 1:
            st.error(f"🚨 Spam (p={p:.3f}, τ={tau:.2f})")
        else:
            st.success(f"✅ Ham (p={p:.3f}, τ={tau:.2f})")

st.caption("Tip: If borderline spam is slipping through, move the threshold slider down (e.g., 0.40).")
