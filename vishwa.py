import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Sample interaction data (User-Product-Rating)
data = {
    'user_id': ['user1', 'user2', 'user3', 'user1', 'user2'],
    'product_id': ['product1', 'product2', 'product3', 'product2', 'product1'],
    'rating': [5, 4, 3, 2, 5]
}

df = pd.DataFrame(data)

# Encode user_id and product_id
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()
df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
df['product_encoded'] = product_encoder.fit_transform(df['product_id'])

# Create a user-item rating matrix
rating_matrix = pd.pivot_table(df, values='rating',
                            index='user_encoded',
                        columns='product_encoded').fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(rating_matrix)

def predict_rating(user_id, product_id):
    try:
        user_idx = user_encoder.transform([user_id])[0]
        product_idx = product_encoder.transform([product_id])[0]
    except ValueError:
        return f"Unknown user or product."

    # Get similar users
    sim_users = user_similarity[user_idx]
    ratings = rating_matrix.iloc[:, product_idx]

    # Weighted sum of ratings
    weighted_ratings = np.dot(sim_users, ratings)
    sim_sum = np.sum(sim_users)

    if sim_sum == 0:
        return "Not enough data to predict."

    predicted_rating = weighted_ratings / sim_sum
    return round(predicted_rating, 2)

# === Test Demo ===
if __name__ == "__main__":
    user = input("Enter user ID (e.g., user1): ")
    product = input("Enter product ID (e.g., product3): ")
    rating = predict_rating(user, product)
    print(f"Predicted Rating for {user} and {product}: {rating}")
