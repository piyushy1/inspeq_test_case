import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# Load dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
df = pd.DataFrame(dataset['train'])
df = df.rename(columns={"sentence": "text", "label": "sentiment"})
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["sentiment_label"] = df["sentiment"].map(label_map)

# Split into reference and production sets
df_ref, df_prod = train_test_split(df, test_size=0.3, random_state=42, stratify=df["sentiment_label"])

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Function to compute mean pooled BERT embeddings
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
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            mean_pool = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            embeddings.append(mean_pool.cpu().numpy())
    return np.vstack(embeddings)

# Get BERT embeddings
X_ref = get_bert_embeddings(df_ref["text"].tolist())
X_prod = get_bert_embeddings(df_prod["text"].tolist())

# Compute IDDS
def compute_idds(X_ref, X_prod):
    p_ref = np.mean(X_ref, axis=0)
    p_prod = np.mean(X_prod, axis=0)
    return wasserstein_distance(p_ref, p_prod)

idds = compute_idds(X_ref, X_prod)
print(f"Input Distribution Drift Score (IDDS): {idds:.4f}")

# Prepare labels
le = LabelEncoder()
y_ref = le.fit_transform(df_ref["sentiment_label"])
y_prod = le.transform(df_prod["sentiment_label"])

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_ref, y_ref)

# Evaluate
y_pred = model.predict(X_prod)
print(classification_report(y_prod, y_pred, target_names=le.classes_))

# Save model and label encoder
joblib.dump(model, "bert_financial_sentiment_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Visualize responsible AI metrics
def plot_metrics():
    cm = pd.crosstab(le.inverse_transform(y_prod), le.inverse_transform(y_pred), rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    metrics = classification_report(y_prod, y_pred, target_names=le.classes_, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose().drop("accuracy")
    metrics_df[["precision", "recall", "f1-score"]].plot(kind='bar', figsize=(10, 6))
    plt.title("Precision, Recall, F1-Score per Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

plot_metrics()

