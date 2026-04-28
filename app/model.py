import time
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, KNNWithZScore, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from preprocess import load_and_build_ratings

def build_surprise_dataset(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "item_id", "rating"]],
        reader
    )
    return data

def train_and_evaluate(data):
    algos = {
        "KNNBasic":    KNNBasic(k=10, sim_options={'name': 'cosine', 'user_based': True}),
        "KNNWithMeans": KNNWithMeans(k=10, sim_options={'name': 'pearson', 'user_based': True}),
        "KNNWithZScore": KNNWithZScore(k=10, sim_options={'name': 'cosine', 'user_based': True}),
        "SVD":         SVD(n_factors=50, n_epochs=20, random_state=42),
    }

    results = {}
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    for name, algo in algos.items():
        t0 = time.time()
        algo.fit(trainset)
        train_time = time.time() - t0

        t1 = time.time()
        predictions = algo.test(testset)
        pred_time = time.time() - t1

        rmse = accuracy.rmse(predictions, verbose=False)
        mae  = accuracy.mae(predictions, verbose=False)

        results[name] = {
            "algo": algo,
            "rmse": rmse,
            "mae":  mae,
            "train_time": train_time,
            "pred_time":  pred_time,
        }
        print(f"[{name}] RMSE={rmse:.4f} | MAE={mae:.4f} | Train={train_time:.3f}s | Pred={pred_time:.4f}s")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n✅ Best algo: {best_name} (RMSE={results[best_name]['rmse']:.4f})")

    return results[best_name]["algo"], results

def get_top_n_recommendations(algo, trainset, user_id, n=3):
    all_items = [trainset.to_raw_iid(iid) for iid in trainset.all_items()]

    try:
        uid_inner = trainset.to_inner_uid(user_id)
        seen = [trainset.to_raw_iid(iid) for iid, _ in trainset.ur[uid_inner]]
    except ValueError:
        seen = []

    unseen = [iid for iid in all_items if iid not in seen and iid != user_id]
    predictions = [algo.predict(user_id, iid) for iid in unseen]
    predictions.sort(key=lambda x: x.est, reverse=True)

    return [(pred.iid, pred.est) for pred in predictions[:n]]

def save_model(algo, trainset, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump((algo, trainset), f)
    print(f"✅ Model saved → {path}")

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)