'''
/**
 * @author Piyush Yadavb
 * @email [mail@piyush-yadav.com]
 * @create date 2025-04-08 23:33:18
 * @modify date 2025-04-08 23:33:18
 * @desc [description]
 */
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    brier_score_loss
)
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

# 1. Load Real-World Financial Dataset
# Example: 'financial_phrasebank' from HuggingFace (used in financial NLP tasks)
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Convert to DataFrame
df = pd.DataFrame(dataset['train'])
df = df.rename(columns={"sentence": "text", "label": "risk_tag"})

# For simplicity, let's make this a binary classification (0: non-risk, 1: risk)
df['risk_tag'] = df['risk_tag'].apply(lambda x: 1 if x == 0 else 0)  # you can adjust based on real risk labels

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["risk_tag"], test_size=0.2, random_state=42
)

# 3. Sentence Embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)

X_train_embed = embedder.encode(X_train.tolist(), show_progress_bar=True)
X_test_embed = embedder.encode(X_test.tolist(), show_progress_bar=True)

# 4. Train Classifier with Calibrated Confidence
base_clf = LogisticRegression(max_iter=1000)
calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train_embed, y_train)
# Save the model
import joblib
joblib.dump(calibrated_clf, "inspeq_casestudy/rai_ml_models/ccs_financial_phrase_bank/financial_phrasebank_model.joblib")

# 5. Predictions
y_pred = calibrated_clf.predict(X_test_embed)
y_proba = calibrated_clf.predict_proba(X_test_embed)[:, 1]

# 6. Responsible AI Metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
roc_auc = roc_auc_score(y_test, y_proba)
brier_score = brier_score_loss(y_test, y_proba)

# Confidence Calibration Summary
confidences = np.max(calibrated_clf.predict_proba(X_test_embed), axis=1)
avg_confidence = np.mean(confidences)

# 7. Output Results
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("🔍 Responsible AI Metrics:")
print(f" - Precision: {precision:.4f}")
print(f" - Recall:    {recall:.4f}")
print(f" - F1 Score:  {f1:.4f}")
print(f" - ROC AUC:   {roc_auc:.4f}")
print(f" - Brier Score (Confidence Calibration): {brier_score:.4f}")
print(f" - Avg. Calibrated Confidence: {avg_confidence:.4f}")
