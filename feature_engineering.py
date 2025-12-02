"""Feature engineering for candidate fit score model."""

from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import normalize_text, extract_years_of_experience


# simple global skill list – this can easily be extended
SKILL_KEYWORDS: List[str] = [
    # languages
    "python", "java", "javascript", "typescript", "c", "c++",
    # web / frontend
    "html", "css", "react", "angular", "frontend",
    # backend / frameworks
    "django", "flask", "spring", "springboot", "spring-boot",
    "node", "node.js", "nodejs", "express",
    # data / ml
    "sql", "mysql", "postgresql", "mongodb", "pandas", "numpy",
    "scikit-learn", "sklearn", "machine", "learning", "ml",
    # devops
    "docker", "kubernetes", "ci/cd", "jenkins", "github", "linux",
    # mobile
    "android", "kotlin"
]


def _collect_skills(text: str) -> List[str]:
    text_lower = str(text).lower()
    found: List[str] = []
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.append(skill)
    # keep unique skills
    return sorted(set(found))


def build_vectorizer(corpus: pd.Series) -> TfidfVectorizer:
    """Create and fit a TF‑IDF vectorizer on the combined corpus."""
    vectorizer = TfidfVectorizer(max_features=700)
    vectorizer.fit(corpus.values.astype("U"))
    return vectorizer


def make_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> Tuple[np.ndarray, List[Dict]]:
    """Transform dataframe into numeric feature matrix.

    Returns X (features) and an info list that we can later use
    when explaining predictions (matched skills, similarity scores, etc.).
    """
    df = df.copy()

    # basic cleaning
    df["resume_clean"] = df["resume_text"].apply(normalize_text)
    df["jd_clean"] = df["job_description"].apply(normalize_text)

    # tf‑idf vectors
    resume_vec = vectorizer.transform(df["resume_clean"].values.astype("U"))
    jd_vec = vectorizer.transform(df["jd_clean"].values.astype("U"))

    # cosine similarity for each pair
    sim_matrix = cosine_similarity(resume_vec, jd_vec)
    cos_sim = sim_matrix.diagonal()

    # token‑level overlaps
    resume_tokens = df["resume_clean"].str.split()
    jd_tokens = df["jd_clean"].str.split()

    overlap_counts: List[int] = []
    jaccard_scores: List[float] = []
    for r_tok, j_tok in zip(resume_tokens, jd_tokens):
        r_set, j_set = set(r_tok), set(j_tok)
        inter = r_set & j_set
        union = r_set | j_set
        overlap_counts.append(len(inter))
        jaccard_scores.append(len(inter) / len(union) if union else 0.0)

    # length features
    resume_len = resume_tokens.apply(len)
    jd_len = jd_tokens.apply(len)
    len_diff = (resume_len - jd_len).abs()
    len_ratio = resume_len / jd_len.replace(0, 1)

    # skill based features
    resume_skills = df["resume_text"].apply(_collect_skills)
    jd_skills = df["job_description"].apply(_collect_skills)

    common_skill_counts: List[int] = []
    skill_jaccard_scores: List[float] = []
    for r_skills, j_skills in zip(resume_skills, jd_skills):
        r_set, j_set = set(r_skills), set(j_skills)
        inter = r_set & j_set
        union = r_set | j_set
        common_skill_counts.append(len(inter))
        skill_jaccard_scores.append(len(inter) / len(union) if union else 0.0)

    # years of experience
    resume_years = df["resume_text"].apply(extract_years_of_experience)
    jd_years = df["job_description"].apply(extract_years_of_experience)
    years_diff = (resume_years - jd_years).abs()

    # assemble numeric features
    features = np.vstack(
        [
            cos_sim,
            np.array(overlap_counts),
            np.array(jaccard_scores),
            resume_len.values,
            jd_len.values,
            len_diff.values,
            len_ratio.values,
            np.array(common_skill_counts),
            np.array(skill_jaccard_scores),
            resume_years.values,
            jd_years.values,
            years_diff.values,
        ]
    ).T

    # extra info used later for explanation
    info: List[Dict] = []
    for i in range(len(df)):
        info.append(
            {
                "cosine_sim": float(cos_sim[i]),
                "jaccard_tokens": float(jaccard_scores[i]),
                "resume_len": int(resume_len.iloc[i]),
                "jd_len": int(jd_len.iloc[i]),
                "length_diff": int(len_diff.iloc[i]),
                "resume_skills": resume_skills.iloc[i],
                "jd_skills": jd_skills.iloc[i],
                "common_skills": sorted(
                    set(resume_skills.iloc[i]) & set(jd_skills.iloc[i])
                ),
                "skill_jaccard": float(skill_jaccard_scores[i]),
                "resume_years": int(resume_years.iloc[i]),
                "jd_years": int(jd_years.iloc[i]),
                "years_diff": int(years_diff.iloc[i]),
            }
        )

    return features, info
