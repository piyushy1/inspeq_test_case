'''
/**
 * @author Piyush Yadavb
 * @email [mail@piyush-yadav.com]
 * @create date 2025-04-08 23:51:54
 * @modify date 2025-04-08 23:51:54
 * @desc [description]
 */
'''

import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer
import joblib
import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Synonym replacer using WordNet (CPU-friendly)
syn_aug = naw.SynonymAug(aug_src='wordnet')

# Example
original_text = "The auditor has expressed concern over accounting practices."
augmented_texts = [syn_aug.augment(original_text) for _ in range(5)]

# Flatten the list of lists into a single list of strings
augmented_texts = [text for sublist in augmented_texts for text in sublist]

print("Original:", original_text)
print("Perturbations:")
for text in augmented_texts:
    print("-", text)

# 3. Sentence Embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# load pretarianed model
calibrated_clf = joblib.load('inspeq_casestudy/rai_ml_models/ccs_financial_phrase_bank/financial_phrasebank_model.joblib')

# Embed augmented texts
augmented_vecs = model.encode(augmented_texts)
probs_augmented = calibrated_clf.predict_proba(augmented_vecs)[:, 1]

# Also get original prediction
orig_vec = model.encode([original_text])
orig_prob = calibrated_clf.predict_proba(orig_vec)[0][1]

# Compute PSS
import numpy as np

std_dev = np.std(probs_augmented)
pss = 1 - std_dev

print(f"Original prediction: {orig_prob:.4f}")
print(f"Augmented predictions: {probs_augmented}")
print(f"Perturbation Sensitivity Score (PSS): {pss:.4f}")

