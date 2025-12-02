"""Flask API for serving candidate fit predictions (bonus part)."""

import pickle
from typing import Dict

import pandas as pd
from flask import Flask, jsonify, request

from feature_engineering import make_features

app = Flask(__name__)


def load_artifact(path: str = "candidate_fit_advanced.pkl") -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


ARTIFACT = None


def get_artifact() -> Dict:
    global ARTIFACT
    if ARTIFACT is None:
        ARTIFACT = load_artifact()
    return ARTIFACT


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json() or {}
    resume_text = payload.get("resume", "")
    jd_text = payload.get("job_description", "")

    if not resume_text or not jd_text:
        return (
            jsonify(
                {
                    "error": "Both 'resume' and 'job_description' must be provided."
                }
            ),
            400,
        )

    artifact = get_artifact()
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    df = pd.DataFrame(
        [
            {
                "resume_text": resume_text,
                "job_description": jd_text,
            }
        ]
    )

    X, info_list = make_features(df, vectorizer)
    info = info_list[0]

    raw_score = float(model.predict(X)[0])
    score = max(0.0, min(100.0, raw_score))

    response = {
        "fit_score": round(score, 2),
        "cosine_similarity": round(info["cosine_sim"], 3),
        "token_overlap_jaccard": round(info["jaccard_tokens"], 3),
        "common_skills": info["common_skills"],
        "skill_jaccard": round(info["skill_jaccard"], 3),
        "resume_years": int(info["resume_years"]),
        "jd_years": int(info["jd_years"]),
        "years_diff": int(info["years_diff"]),
    }

    return jsonify(response)


if __name__ == "__main__":
    # simple dev server, good enough for the assignment
    app.run(debug=True)
