import sys
sys.setrecursionlimit(5000)
import joblib
import m2cgen as m2c
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# --- Config ---
model_path = 'inspeq_casestudy/rai_ml_models/input_distribution_drift_score_lr_svm/bert_financial_sentiment_model.pkl'
output_c_path = 'inspeq_casestudy/low_latency_optimization/logreg_model.c'
output_js_path = 'inspeq_casestudy/low_latency_optimization/logreg_model.js'
template_file = 'inspeq_casestudy/low_latency_optimization/model_info_template.j2'

# --- Metadata ---
author_name = "Piyush Yadav"
creation_date = datetime.now().strftime("%Y-%m-%d")
model_type = "Logistic Regression"
model_accuracy = "N/A"  # Replace with real accuracy if available

# --- Load model with joblib ---
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"❌ Error loading model with joblib: {e}")
    exit()

# --- Optional: Check model type ---
print(f"Loaded model: {type(model)}")

# --- Export using m2cgen ---
try:
    c_code = m2c.export_to_c(model)
    js_code = m2c.export_to_javascript(model)
except Exception as e:
    print(f"❌ Error during model code generation: {e}")
    exit()

# --- Load Jinja template ---
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template(template_file)

# --- Render and Save C Code ---
rendered_c_code = template.render(
    author=author_name,
    date_created=creation_date,
    model_type=model_type,
    accuracy=model_accuracy,
    language='c'
) + "\n\n" + c_code

with open(output_c_path, 'w') as f:
    f.write(rendered_c_code)
print(f"✅ C code saved to: {output_c_path}")

# --- Render and Save JavaScript Code ---
rendered_js_code = template.render(
    author=author_name,
    date_created=creation_date,
    model_type=model_type,
    accuracy=model_accuracy,
    language='javascript'
) + "\n\n" + js_code

with open(output_js_path, 'w') as f:
    f.write(rendered_js_code)
print(f"✅ JavaScript code saved to: {output_js_path}")
