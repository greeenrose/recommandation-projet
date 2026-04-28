import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import ast

from model import load_model, get_top_n_recommendations, save_model, build_surprise_dataset, train_and_evaluate
from preprocess import load_and_build_ratings

app = FastAPI(
    title="Système de Recommandation Étudiants",
    description="API de recommandation de coéquipiers basée sur KNN (Surprise)",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_etudiants1.csv")

algo        = None
trainset    = None
df_students = None

@app.on_event("startup")
async def startup_event():
    global algo, trainset, df_students

    df_students, ratings_df = load_and_build_ratings(DATA_PATH)

    if os.path.exists(MODEL_PATH):
        algo, trainset = load_model(MODEL_PATH)
        print("✅ Model loaded from pkl")
    else:
        print("⚙️  Training model...")
        data = build_surprise_dataset(ratings_df)
        algo, _ = train_and_evaluate(data)
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        save_model(algo, trainset, MODEL_PATH)

# ── Pydantic models ──────────────────────────────────────────

class RecommendationItem(BaseModel):
    student_id:      int
    student_name:    str
    predicted_score: float
    competences:     List[str]
    communautes:     List[str]

class RecommendationResponse(BaseModel):
    user_id:         int
    user_name:       str
    recommendations: List[RecommendationItem]
    algo_used:       str

# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "API Recommandation — go to /docs for Swagger UI"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": algo is not None}

@app.get("/students")
def list_students():
    return {"students": df_students[["ID_Étudiant", "Nom"]].to_dict("records")}

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, n: int = 3):
    if user_id not in df_students["ID_Étudiant"].values:
        raise HTTPException(status_code=404, detail=f"Étudiant {user_id} introuvable")

    recs = get_top_n_recommendations(algo, trainset, user_id=user_id, n=n)

    rec_list = []
    for rec_id, score in recs:
        row = df_students[df_students["ID_Étudiant"] == rec_id].iloc[0]
        comp = row["Compétences"]   if isinstance(row["Compétences"], list)  else []
        comm = row["Communautés"]   if isinstance(row["Communautés"], list)  else []
        rec_list.append(RecommendationItem(
            student_id=int(rec_id),
            student_name=row["Nom"],
            predicted_score=round(score, 3),
            competences=comp,
            communautes=comm
        ))

    user_name = df_students[df_students["ID_Étudiant"] == user_id]["Nom"].values[0]

    return RecommendationResponse(
        user_id=user_id,
        user_name=user_name,
        recommendations=rec_list,
        algo_used=type(algo).__name__
    )