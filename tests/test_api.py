import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Set data path before importing app
os.environ["DATA_PATH"] = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_etudiants1.csv")

from preprocess import load_and_build_ratings
from model import build_surprise_dataset, train_and_evaluate, get_top_n_recommendations

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_etudiants1.csv")

def test_data_loads():
    df, ratings_df = load_and_build_ratings(DATA_PATH)
    assert len(df) > 0
    assert len(ratings_df) > 0

def test_model_trains():
    df, ratings_df = load_and_build_ratings(DATA_PATH)
    data = build_surprise_dataset(ratings_df)
    algo, results = train_and_evaluate(data)
    assert algo is not None
    assert "KNNBasic" in results

def test_recommendations():
    df, ratings_df = load_and_build_ratings(DATA_PATH)
    data = build_surprise_dataset(ratings_df)
    algo, _ = train_and_evaluate(data)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    recs = get_top_n_recommendations(algo, trainset, user_id=1, n=3)
    assert len(recs) > 0