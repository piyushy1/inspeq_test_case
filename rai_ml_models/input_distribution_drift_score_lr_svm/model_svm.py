import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance
from transformers import BertTokenizer, BertModel

# 1. Load dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
df = pd.DataFrame(dataset['train'])
df = df.rename(columns={"sentence": "text", "label": "sentiment"})
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["sentiment_label"] = df["sentiment"].map(label_map)

# 2. Train-test split
df_ref, df_prod = train_test_split(df, test_size=0.3, random_state=42, stratify=df["sentiment_label"])

# 3. BERT model setup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            mean_pool = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            embeddings.append(mean_pool.cpu().numpy())
    return np.vstack(embeddings)

# 4. Generate embeddings
X_ref = get_bert_embeddings(df_ref["text"].tolist())
X_prod = get_bert_embeddings(df_prod["text"].tolist())

# 5. Compute IDDS
def compute_idds(X_ref, X_prod):
    p_ref = np.mean(X_ref, axis=0)
    p_prod = np.mean(X_prod, axis=0)
    return wasserstein_distance(p_ref, p_prod)

idds = compute_idds(X_ref, X_prod)
print(f"\n📊 Input Distribution Drift Score (IDDS): {idds:.4f}\n")

# 6. Train SVM classifier
le = LabelEncoder()
y_ref = le.fit_transform(df_ref["sentiment_label"])
y_prod = le.transform(df_prod["sentiment_label"])

svm_model = SVC(kernel="linear", probability=True)  # You can also try 'rbf'
svm_model.fit(X_ref, y_ref)
y_pred = svm_model.predict(X_prod)

# 7. Classification report
print("\n🧪 Classification Report (SVM):\n")
print(classification_report(y_prod, y_pred, target_names=le.classes_))

# 8. Save model and encoder
joblib.dump(svm_model, "inspeq_casestudy/rai_ml_models/input_distribution_drift_score_lr_svm/bert_svm_financial_sentiment_model.pkl")
joblib.dump(le, "inspeq_casestudy/rai_ml_models/input_distribution_drift_score_lr_svm/label_encoder.pkl")

# 9. Evaluation visualizations
def plot_metrics():
    cm = pd.crosstab(le.inverse_transform(y_prod), le.inverse_transform(y_pred), rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - SVM")
    plt.show()

    metrics = classification_report(y_prod, y_pred, target_names=le.classes_, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose().drop("accuracy")
    metrics_df[["precision", "recall", "f1-score"]].plot(kind='bar', figsize=(10, 6))
    plt.title("Precision, Recall, F1-Score per Class - SVM")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

plot_metrics()
