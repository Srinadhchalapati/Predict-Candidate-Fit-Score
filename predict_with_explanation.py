"""CLI tool to score a single candidate and explain the match factors."""

import json
import pickle
from typing import Dict

import pandas as pd

from feature_engineering import make_features
from preprocessing import normalize_text


def load_artifact(path: str = "candidate_fit_advanced.pkl") -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def explain_prediction(resume_text: str, jd_text: str) -> Dict:
    artifact = load_artifact()
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

    # build human‑readable explanation
    explanation = {
        "fit_score": round(score, 2),
        "cosine_similarity": round(info["cosine_sim"], 3),
        "token_overlap_jaccard": round(info["jaccard_tokens"], 3),
        "length_diff": info["length_diff"],
        "resume_length": info["resume_len"],
        "jd_length": info["jd_len"],
        "resume_years": info["resume_years"],
        "jd_years": info["jd_years"],
        "years_diff": info["years_diff"],
        "common_skills": info["common_skills"],
        "skill_jaccard": round(info["skill_jaccard"], 3),
    }

    # add short verbal summary
    summary_parts = []
    if explanation["common_skills"]:
        summary_parts.append(
            f"Good skill match: {', '.join(explanation['common_skills'])}."
        )
    else:
        summary_parts.append("Very few explicit skills in common.")

    if explanation["cosine_similarity"] > 0.5:
        summary_parts.append("Overall text of resume and JD is quite similar.")
    else:
        summary_parts.append("Text similarity between resume and JD is moderate/low.")

    if explanation["years_diff"] == 0:
        summary_parts.append("Years of experience appears to match the requirement.")
    elif explanation["resume_years"] > explanation["jd_years"]:
        summary_parts.append("Candidate seems slightly more experienced than the JD.")
    else:
        summary_parts.append("Candidate seems less experienced than the JD mentions.")

    explanation["summary"] = " ".join(summary_parts)
    return explanation


if __name__ == "__main__":
    print("Advanced Candidate Fit Score - Demo\n")

    resume = input("Paste candidate resume text:\n> ")
    jd = input("\nPaste job description text:\n> ")

    result = explain_prediction(resume, jd)

    print("\n---------------- Result ----------------")
    print("Fit score:", result["fit_score"], "/ 100")
    print("Common skills:", ", ".join(result["common_skills"]) or "None")
    print("Cosine similarity:", result["cosine_similarity"])
    print("Token Jaccard:", result["token_overlap_jaccard"])
    print("Skill Jaccard:", result["skill_jaccard"])
    print("Years (resume / jd):", result["resume_years"], "/", result["jd_years"])
    print("Summary:", result["summary"])
    print("----------------------------------------\n")
