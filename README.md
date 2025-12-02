# Predict Candidate Fit Score – Advanced ML Solution (No Deep Learning)

This project implements an **advanced machine learning pipeline** that predicts
how well a candidate's resume fits a job description. The model outputs a
numeric fit score between **0 and 100** and also provides **explanations of the
key matching factors**.

The solution follows the requirements from the task description:

## 1. Technologies

- Python
- NumPy, pandas
- scikit-learn
- Flask (for the Web API)

## 2. Approach Overview

1. **Data Processing**
   - Parse resume and job description text from CSV.
   - Normalize text (lower‑casing, removal of punctuation and stopwords).
   - Light rule‑based extraction of years of experience.

2. **Feature Engineering**
   - TF‑IDF vectorization of both resume and job description text.
   - Cosine similarity between resume TF‑IDF and JD TF‑IDF.
   - Token overlap and token‑level Jaccard similarity.
   - Length features (individual lengths, absolute difference, length ratio).
   - Skill‑based features:
     - Detect skills from a small curated list (Python, Java, React, SQL, Docker, etc.).
     - Number of common skills.
     - Skill‑level Jaccard similarity.
   - Years of experience features:
     - Years mentioned in resume.
     - Years mentioned in job description.
     - Absolute difference between them.

3. **Model Development**
   - The numeric features are fed into a
     **GradientBoostingRegressor** (from scikit‑learn).
   - Target variable is the manual fit score column in the dataset.
   - Train/validation split (75% / 25%).

4. **Evaluation**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² score
   - Metrics are printed by `train_advanced_model.py`.

5. **Output**
   - `predict_with_explanation.py` loads the trained model and prints:
     - Fit score (0–100).
     - Common skills.
     - Similarity scores.
     - Short textual summary explaining the match.

6. **Bonus: Web API**
   - `advanced_api.py` exposes a simple Flask API:
     - `GET /health` – health‑check.
     - `POST /predict` – accepts JSON with `resume` and `job_description`.
     - Returns fit score + key matching factors as JSON.

## 3. File Overview

- `data/sample_data.csv` – small demo dataset with 10 candidate/JD pairs.
- `preprocessing.py` – text normalization and years‑of‑experience heuristic.
- `feature_engineering.py` – construction of all numeric features.
- `train_advanced_model.py` – trains the GradientBoostingRegressor and saves `candidate_fit_advanced.pkl`.
- `predict_with_explanation.py` – CLI tool to score a single candidate and explain the match.
- `advanced_api.py` – Flask API endpoint to serve predictions (bonus).
- `requirements.txt` – Python dependencies.

## 4. How to Run

```bash
# 1) install dependencies
pip install -r requirements.txt

# 2) train the model
python train_advanced_model.py
# this reads data/sample_data.csv,
# trains the model and saves candidate_fit_advanced.pkl

# 3) run a single prediction from the command line
python predict_with_explanation.py
# paste resume text and job description when asked

# 4) (optional) start the API
python advanced_api.py
# POST a JSON body to /predict:
# {
#   "resume": "some resume text",
#   "job_description": "some JD text"
# }
```

## 5. Notes

- The project is kept intentionally small and readable, so it can be
  explained easily in an interview.
- At the same time it uses a **non‑trivial feature set** and a
  **strong gradient boosting model**, which satisfies the "advanced"
  requirement without relying on deep learning libraries.
"# Predict-Candidate-Fit-Score" 
