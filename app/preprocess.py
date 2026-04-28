import pandas as pd
import ast

def load_and_build_ratings(filepath):
    df = pd.read_csv(filepath)

    for col in ["Coéquipiers", "Communautés", "Compétences", "Centres_d'Intérêt"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    ratings = []

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if row_i["ID_Étudiant"] == row_j["ID_Étudiant"]:
                continue

            score = 0

            if row_j["ID_Étudiant"] in row_i["Coéquipiers"]:
                score += 2

            comm_commune = set(row_i["Communautés"]) & set(row_j["Communautés"])
            score += len(comm_commune)

            comp_commune = set(row_i["Compétences"]) & set(row_j["Compétences"])
            score += min(len(comp_commune), 2)

            rating = min(max(score, 0), 5)

            ratings.append({
                "user_id": row_i["ID_Étudiant"],
                "item_id": row_j["ID_Étudiant"],
                "rating": float(rating)
            })

    ratings_df = pd.DataFrame(ratings)
    ratings_df = ratings_df[ratings_df["rating"] > 0]

    return df, ratings_df