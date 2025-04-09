from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from metrics import evaluate_ccs_score, evaluate_pss_score
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    text: str  # For the text input

# Load the pre-trained model and vectorizer once at the start
model = joblib.load('/Users/piyush/Desktop/Codes/inspeq_casestudy/rai_dashboard/models/ccs_brier_financial_phrasebank_model.joblib')
# Load the vectorizer (if applicable)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
vectorizer = SentenceTransformer(model_name)

@app.post("/evaluate/")
async def evaluate(input_data: InputData):
    text = input_data.text
    print(f"Received text: {text}")

    # For demo purposes, we'll assume a ground truth label is 1
    y_test = [1]  # Dummy label (you can replace it with real labels if available)
    
    # Call the function to calculate RAI metrics (CCS Score)
    ccs_metrics = evaluate_ccs_score(model, vectorizer, [text], y_test)
    pss_metrics = evaluate_pss_score(model, vectorizer, text)

    return {
        **ccs_metrics,
        **pss_metrics
    }

    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)