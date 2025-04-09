from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Import model evaluation functions
from model import (
    evaluate_brier_score,
    evaluate_ccs_score,
    evaluate_model_drift,
    evaluate_feature_drift,
    evaluate_input_distribution_drift
)

app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    text: str  # For the text input

# Load the models once at the start
brier_adult_income_model = joblib.load('inspeq_casestudy/rai_dashboard/models/brier_adult_income_model.joblib')
ccs_brier_financial_phrasebank_model = joblib.load('inspeq_casestudy/rai_dashboard/models/ccs_brier_financial_phrasebank_model.joblib')
finance_hate_model = joblib.load('inspeq_casestudy/rai_dashboard/models/finance_hate_model.joblib')
idd_bert_financial_sentiment_model = joblib.load('inspeq_casestudy/rai_dashboard/models/idd_bert_financial_sentiment_model.pkl')
idd_bert_svm_financial_sentiment_model = joblib.load('inspeq_casestudy/rai_dashboard/models/idd_bert_svm_financial_sentiment_model.pkl')


# Load the vectorizer (if applicable)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
vectorizer = SentenceTransformer(model_name)
# vectorizer = joblib.load('inspeq_casestudy/rai_dashboard/models/vectorizer.pkl')

@app.post("/evaluate/")
async def evaluate(input_data: InputData):
    text = input_data.text

    # Example ground truth label if required for evaluation (you can modify this according to the task)
    y_true = 1  # This can be dynamic, or passed from the front-end if necessary

    # Evaluate each RAI metric using the loaded models
    brier_metrics = evaluate_brier_score(brier_adult_income_model, vectorizer, text, y_true)
    ccs_metrics = evaluate_ccs_score(ccs_brier_financial_phrasebank_model, vectorizer, text, y_true)
    model_drift = evaluate_model_drift(brier_adult_income_model, finance_hate_model, vectorizer, text)
    feature_drift = evaluate_feature_drift(idd_bert_financial_sentiment_model, vectorizer, text, 'original_feature', 'new_feature')
    input_drift = evaluate_input_distribution_drift(idd_bert_svm_financial_sentiment_model, vectorizer, text, ['reference_data_1', 'reference_data_2'])

    # Combine all metrics into a single response
    return {
        "brier_score": brier_metrics,
        "ccs_score": ccs_metrics,
        "model_drift": model_drift,
        "feature_drift": feature_drift,
        "input_distribution_drift": input_drift
    }
