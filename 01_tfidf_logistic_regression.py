"""
Slides 13-23: Spam Classifier — TF-IDF + Logistic Regression + Gradient Descent
--------------------------------------------------------------------------------
Implements, from scratch, exactly the formulas from the PDF:

    TF(t,d)   = count(t in d) / total_words(d)
    IDF(t)    = log((N+1)/(df(t)+1)) + 1
    TFIDF     = TF * IDF

    f(x) = w1*x + w0        (linear score)
    P(Spam) = sigmoid(w.x + b) = 1 / (1 + e^-(wx+b))
    L(y, y_hat) = -[y*log(y_hat) + (1-y)*log(1-y_hat)]     (Binary Cross-Entropy)
    dL/dw = (y_hat - y) * x                                (gradient)
    w_new = w - alpha * dL/dw                              (gradient descent update)

No sklearn / no external ML libs — pure numpy, so every line traces back to a
formula on the slide.
"""

import re
import math
import numpy as np


# ---------------------------------------------------------------------------
# 1. Toy dataset (email text, label). 1 = Spam, 0 = Ham.
# ---------------------------------------------------------------------------
EMAILS = [
    ("You have WON a free prize! Claim your jackpot now", 1),
    ("Free entry to win a lottery prize, click now", 1),
    ("Congratulations winner, claim your free cash prize", 1),
    ("Limited offer, free gift, click the link now", 1),
    ("Hey, are we still meeting for lunch tomorrow?", 0),
    ("Please review the attached report before the meeting", 0),
    ("Can you send me the notes from class today?", 0),
    ("Let's catch up over coffee this weekend", 0),
]


def tokenize(text: str):
    return re.findall(r"[a-zA-Z]+", text.lower())


# ---------------------------------------------------------------------------
# 2. TF-IDF, implemented exactly per the slide formulas
# ---------------------------------------------------------------------------
def build_vocab(docs_tokens):
    vocab = sorted({tok for toks in docs_tokens for tok in toks})
    return {tok: i for i, tok in enumerate(vocab)}


def term_frequency(tokens):
    """TF(t, d) = count(t in d) / total_words(d)"""
    tf = {}
    total = len(tokens)
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    return {tok: c / total for tok, c in tf.items()}


def inverse_document_frequency(docs_tokens, vocab):
    """IDF(t) = log((N+1)/(df(t)+1)) + 1"""
    N = len(docs_tokens)
    idf = {}
    for term in vocab:
        df = sum(1 for toks in docs_tokens if term in toks)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf


def tfidf_vector(tokens, vocab, idf):
    """TFIDF(t,d) = TF(t,d) * IDF(t) -> dense vector over vocab"""
    tf = term_frequency(tokens)
    vec = np.zeros(len(vocab))
    for tok, freq in tf.items():
        if tok in vocab:
            vec[vocab[tok]] = freq * idf[tok]
    return vec


# ---------------------------------------------------------------------------
# 3. Logistic regression: sigmoid, BCE loss, gradient descent
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true, y_pred, eps=1e-9):
    """L(y, y_hat) = -[y*log(y_hat) + (1-y)*log(1-y_hat)]"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_logistic_regression(X, y, lr=0.5, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)   # weight vector
    b = 0.0                    # bias

    for epoch in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)

        loss = bce_loss(y, y_hat)

        # dL/dw = (y_hat - y) . x   (averaged over the batch)
        grad_w = X.T @ (y_hat - y) / n_samples
        grad_b = np.mean(y_hat - y)

        # w_new = w - alpha * dL/dw
        w -= lr * grad_w
        b -= lr * grad_b

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} | loss = {loss:.4f}")

    return w, b


# ---------------------------------------------------------------------------
# 4. Run the whole pipeline
# ---------------------------------------------------------------------------
def main():
    texts = [t for t, _ in EMAILS]
    labels = np.array([lab for _, lab in EMAILS], dtype=float)

    docs_tokens = [tokenize(t) for t in texts]
    vocab = build_vocab(docs_tokens)
    idf = inverse_document_frequency(docs_tokens, vocab)

    X = np.array([tfidf_vector(toks, vocab, idf) for toks in docs_tokens])

    print(f"Vocabulary size: {len(vocab)}")
    print("Training logistic regression via gradient descent...\n")
    w, b = train_logistic_regression(X, labels, lr=0.5, epochs=500)

    print("\n--- Predictions on training set ---")
    probs = sigmoid(X @ w + b)
    for text, label, p in zip(texts, labels, probs):
        pred = "SPAM" if p >= 0.5 else "HAM"
        print(f"[{pred:4s} p={p:.3f}] true={int(label)}  \"{text[:50]}\"")

    # Try a brand-new message
    new_text = "Win a free prize now, click here to claim"
    new_vec = tfidf_vector(tokenize(new_text), vocab, idf)
    p_new = sigmoid(new_vec @ w + b)
    print(f"\nNew message: \"{new_text}\"")
    print(f"P(Spam) = {p_new:.3f} -> {'SPAM' if p_new >= 0.5 else 'HAM'}")


if __name__ == "__main__":
    main()
