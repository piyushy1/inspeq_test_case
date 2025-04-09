from sklearn.metrics import brier_score_loss
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, brier_score_loss, classification_report
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer

syn_aug = naw.SynonymAug(aug_src='wordnet')  # WordNet-based synonym augmentations


# 1. Brier Score Evaluation
def evaluate_brier_score(model, vectorizer, x_text, y_true):
    """
    Function to evaluate Brier Score.
    """
    x_vec = vectorizer.encode([x_text])
    prob = model.predict_proba(x_vec)[0][1]
    brier = brier_score_loss([y_true], [prob])
    return {"brier_score": round(brier, 4), "confidence": round(prob, 4)}

# 2. CCS Score Evaluation (Placeholder function)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, brier_score_loss
import numpy as np
import math

def sanitize_float(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None  # or 0.0 or "NaN" based on preference
    return round(value, 4)

def evaluate_ccs_score(model, vectorizer, texts, y_true):
    try:
        # Step 1: Vectorize input text
        X_embed = vectorizer.encode(texts)

        # Step 2: Make predictions
        y_pred = model.predict(X_embed)
        y_proba = model.predict_proba(X_embed)[:, 1]

        # Step 3: Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)
        avg_conf = np.mean(np.max(model.predict_proba(X_embed), axis=1))

        # Step 4: Return safe JSON
        return {
            "precision": sanitize_float(precision),
            "recall": sanitize_float(recall),
            "f1_score": sanitize_float(f1),
            "roc_auc": sanitize_float(roc_auc),
            "brier_score": sanitize_float(brier),
            "avg_confidence": sanitize_float(avg_conf)
        }

    except Exception as e:
        print("🔥 Error in CCS evaluation:", str(e))
        raise


# 3. Drift Evaluation (Model Drift)
def evaluate_model_drift(model_1, model_2, vectorizer, x_text):
    """
    Function to evaluate Model Drift.
    Compare predictions from two models on the same input text.
    """
    prob_1 = model_1.predict_proba(vectorizer.encode([x_text]))[0][1]
    prob_2 = model_2.predict_proba(vectorizer.encode([x_text]))[0][1]
    drift = abs(prob_1 - prob_2)  # Simplified drift measure
    return {"model_drift": round(drift, 4)}

# 4. Feature Drift Evaluation (Placeholder function)
def evaluate_feature_drift(model, vectorizer, x_text, original_features, new_features):
    """
    Function to evaluate Feature Drift.
    Compare feature vectors from original and new data.
    """
    x_vec_original = vectorizer.encode([x_text])
    x_vec_new = vectorizer.encode([new_features])
    drift = np.linalg.norm(x_vec_original - x_vec_new)  # Euclidean distance as a simple drift metric
    return {"feature_drift": round(drift, 4)}

# 5. Input Distribution Drift Evaluation
def evaluate_input_distribution_drift(model, vectorizer, x_text, reference_data):
    """
    Function to evaluate Input Distribution Drift.
    Compare the input text's vectorized representation with the distribution of reference data.
    """
    x_vec = vectorizer.encode([x_text])
    scaler = StandardScaler()
    reference_vectors = [vectorizer.encode([text]) for text in reference_data]
    scaler.fit(reference_vectors)  # Fit the scaler on the reference data
    scaled_input = scaler.transform(x_vec)
    drift = np.linalg.norm(scaled_input - np.mean(reference_vectors, axis=0))
    return {"input_distribution_drift": round(drift, 4)}

def evaluate_pss_score(model, vectorizer, text, n_augmentations=5):
    """
    Evaluates Perturbation Sensitivity Score (PSS) for a given text input.
    """
    syn_aug = naw.SynonymAug(aug_src='wordnet')  # Localize for thread-safety

    augmented_texts = []
    for _ in range(n_augmentations):
        augmented = syn_aug.augment(text)
        if isinstance(augmented, list):
            augmented_texts.extend(augmented)
        else:
            augmented_texts.append(augmented)

    augmented_vecs = vectorizer.encode(augmented_texts)
    orig_vec = vectorizer.encode([text])
    probs_augmented = model.predict_proba(augmented_vecs)[:, 1]
    orig_prob = model.predict_proba(orig_vec)[0][1]
    std_dev = np.std(probs_augmented)
    pss = 1 - std_dev

    return {
        "pss_score": round(pss, 4),
        "original_prob": round(orig_prob, 4),
        "augmented_std": round(std_dev, 4),
        "perturbed_texts": augmented_texts  # Add this line
    }
