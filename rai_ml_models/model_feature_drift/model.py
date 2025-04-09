
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

# --- 1. Synthetic Dataset Generation ---
np.random.seed(42)
n_samples = 1000
texts = [
    f"This is a positive review about product {i}." if np.random.rand() > 0.3 else f"This is a negative comment on service {i}."
    for i in range(n_samples)
]
labels = [1 if "positive" in text else 0 for text in texts]
df = pd.DataFrame({'text': texts, 'label': labels})

# --- 2. Initial Model Training (Time Period 1) ---
train_df_1, test_df_1 = train_test_split(df, test_size=0.2, random_state=42)

vectorizer_1 = TfidfVectorizer()
X_train_1 = vectorizer_1.fit_transform(train_df_1['text'])
y_train_1 = train_df_1['label']
X_test_1 = vectorizer_1.transform(test_df_1['text'])
y_test_1 = test_df_1['label']

model_1 = LogisticRegression(random_state=42)
model_1.fit(X_train_1, y_train_1)

y_pred_1 = model_1.predict(X_test_1)
accuracy_1 = accuracy_score(y_test_1, y_pred_1)
print(f"Initial Model (Time Period 1) Accuracy: {accuracy_1:.4f}")
print("Initial Model (Time Period 1) Classification Report:\n", classification_report(y_test_1, y_pred_1))

# --- 3. Simulating Data Drift Over Time ---
time_periods = 5
drift_steps = 100
drift_data = []

for t in range(2, time_periods + 1):
    print(f"\n--- Simulating Time Period {t} ---")
    # Introduce drift: Gradually change the sentiment or topic
    drifted_texts = []
    drifted_labels = []
    for i in range(drift_steps):
        if np.random.rand() < (t - 1) * 0.15:  # Increasing probability of drift
            # Flip sentiment with some probability
            original_text = texts[np.random.randint(0, n_samples)]
            if "positive" in original_text:
                drifted_texts.append(original_text.replace("positive", "mixed"))
                drifted_labels.append(0)  # Treat mixed as negative for simplicity
            elif "negative" in original_text:
                drifted_texts.append(original_text.replace("negative", "neutral"))
                drifted_labels.append(1)  # Treat neutral as positive for simplicity
            else:
                drifted_texts.append(original_text)
                drifted_labels.append(labels[np.random.randint(0, n_samples)])
        else:
            drifted_texts.append(texts[np.random.randint(0, n_samples)])
            drifted_labels.append(labels[np.random.randint(0, n_samples)])

    drift_df = pd.DataFrame({'text': drifted_texts, 'label': drifted_labels})
    drift_data.append(drift_df)

# --- 4. Evaluating Model Performance Over Time ---
accuracy_over_time = [accuracy_1]
drift_scores_feature = []
drift_scores_output = []

for i, drift_df in enumerate(drift_data):
    time_period = i + 2
    X_drift = vectorizer_1.transform(drift_df['text']) # Use the original vectorizer
    y_drift = drift_df['label']
    y_pred_drift = model_1.predict(X_drift)
    accuracy_drift = accuracy_score(y_drift, y_pred_drift)
    accuracy_over_time.append(accuracy_drift)
    print(f"Model Performance (Time Period {time_period}): Accuracy = {accuracy_drift:.4f}")
    print(f"Model Performance (Time Period {time_period}) Classification Report:\n", classification_report(y_drift, y_pred_drift))

    # --- Feature Drift Detection (Distribution of TF-IDF vectors) ---
    # Extract the feature vectors for the initial test set and the current drifted set
    features_original = X_test_1.toarray().flatten()
    features_drifted = X_drift.toarray().flatten()

    # Calculate Wasserstein distance (Earth Mover's Distance) as a drift metric
    drift_feature = wasserstein_distance(features_original, features_drifted)
    drift_scores_feature.append(drift_feature)
    print(f"Feature Drift (Wasserstein Distance) - Time Period {time_period}: {drift_feature:.4f}")

    # --- Output Drift Detection (Distribution of Predicted Probabilities) ---
    try:
        probs_original = model_1.predict_proba(X_test_1)[:, 1] # Probability of positive class
        probs_drifted = model_1.predict_proba(X_drift)[:, 1]
        drift_output = wasserstein_distance(probs_original, probs_drifted)
        drift_scores_output.append(drift_output)
        print(f"Output Drift (Wasserstein Distance) - Time Period {time_period}: {drift_output:.4f}")
    except AttributeError:
        print("Model does not support predict_proba for output drift calculation.")
        drift_scores_output.append(np.nan)


# --- 5. Visualization of Model Drift Performance ---
time_points = range(1, time_periods + 1)

plt.figure(figsize=(12, 5))

# Plot Accuracy Over Time
plt.subplot(1, 2, 1)
plt.plot(time_points, accuracy_over_time, marker='o')
plt.title('Model Accuracy Over Time')
plt.xlabel('Time Period')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot Drift Scores Over Time
plt.subplot(1, 2, 2)
plt.plot(time_points[1:], drift_scores_feature, marker='o', label='Feature Drift')
plt.plot(time_points[1:], drift_scores_output, marker='x', label='Output Drift')
plt.title('Model Drift Scores Over Time')
plt.xlabel('Time Period')
plt.ylabel('Drift Score (Wasserstein Distance)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 6. Further Visualization (Optional - Distribution Changes) ---
if time_periods > 2:
    plt.figure(figsize=(15, 5))

    # Distribution of TF-IDF features (first few features for visualization)
    plt.subplot(1, 2, 1)
    sns.histplot(X_test_1.toarray()[:, 0], label='Initial (Time 1)', kde=True)
    sns.histplot(X_drift.toarray()[:, 0], label=f'Drifted (Time {time_periods})', color='orange', kde=True)
    plt.title('Distribution of First TF-IDF Feature')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Distribution of Predicted Probabilities
    if hasattr(model_1, 'predict_proba'):
        plt.subplot(1, 2, 2)
        sns.histplot(probs_original, label='Initial (Time 1)', kde=True)
        sns.histplot(probs_drifted, label=f'Drifted (Time {time_periods})', color='orange', kde=True)
        plt.title('Distribution of Predicted Probabilities (Positive Class)')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()