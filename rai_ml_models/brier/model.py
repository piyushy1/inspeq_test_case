'''

/**
 * @author Piyush Yadavb
 * @email [mail@piyush-yadav.com]
 * @create date 2025-04-08 20:52:10
 * @modify date 2025-04-08 20:52:10
 * @desc [This script preprocesses the adult dataset for the Brier case study. It loads the dataset, encodes categorical features using SentenceTransformer, and saves the processed data.]
 * @note [This script is part of the Brier case study for the Inspeq project. It uses the SentenceTransformer model to encode categorical features into embeddings, which are then used for training a logistic regression model. The logistic regression model is used to predict the probability of income based on various features. The Brier score is calculated to evaluate the model's performance.]
 */
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ---------------------------
# Load dataset
def load_data(data_path:str):
    df = pd.read_csv(data_path,
                    header=None, names=[
                        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                        'marital-status', 'occupation', 'relationship', 'race', 'sex',
                        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

    # Drop missing values & strip whitespace
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.replace('?', pd.NA).dropna()

    # Label encode income
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})

    return df

# ---------------------------
# Encode categorical features
def encode_features(df:pd.DataFrame):
    # Split into train and test sets
    X = df.drop('income', axis=1)
    y = df['income']
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def tabular_to_text(row):
        return ', '.join([f"{col}: {val}" for col, val in row.items()])

    X_text = X.apply(tabular_to_text, axis=1)
    X_embeddings = model.encode(X_text.tolist(), convert_to_numpy=True)
    return X_embeddings[1:], y[1:]

# ---------------------------
# train_model

def train_save_model(X_embeddings, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    # Save the model
    joblib.dump(clf, "inspeq_casestudy/rai_ml_models/brier/adult_model.joblib")

    # Predict probability
    y_probs = clf.predict_proba(X_test)[:, 1]

    # Calculate Brier Score
    brier = brier_score_loss(y_test, y_probs)
    print(f"Brier Score: {brier:.4f}")


data_path ='inspeq_casestudy/rai_ml_models/brier/adult.csv'

data = load_data(data_path)
X_embeddings, y = encode_features(data)
train_save_model(X_embeddings, y)
