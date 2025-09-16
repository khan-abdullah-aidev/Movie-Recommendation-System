import numpy as np

# User-Item rating matrix
ratings = np.array([
    [5, 4, 0, 0],  # User 1
    [4, 5, 0, 0],  # User 2
    [0, 0, 5, 4],  # User 3
    [0, 0, 4, 5],  # User 4
    [5, 0, 4, 0],  # User 5 (bridge user: connects both sides)
])

# Step 1: Cosine similarity between users
def cosine_similarity(u, v):
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 0
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

n_users = ratings.shape[0]
similarity_matrix = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        similarity_matrix[i, j] = cosine_similarity(ratings[i], ratings[j])

print("Cosine Similarity Matrix:\n", similarity_matrix)

# Step 2: Predict ratings for unrated movies (weighted average of neighbors)
def predict_scores(user_index):
    user_ratings = ratings[user_index]
    scores = np.zeros(user_ratings.shape)
    
    for movie in range(len(user_ratings)):
        if user_ratings[movie] == 0:  # only predict unrated movies
            num = 0
            den = 0
            for other in range(n_users):
                if other != user_index and ratings[other][movie] > 0:
                    sim = similarity_matrix[user_index][other]
                    num += sim * ratings[other][movie]
                    den += abs(sim)
            scores[movie] = num / den if den != 0 else 0
    
    # Fallback: if all 0, recommend the most popular movie
    if np.all(scores == 0):
        avg_ratings = np.true_divide(ratings.sum(axis=0), (ratings != 0).sum(axis=0))
        avg_ratings[np.isnan(avg_ratings)] = 0
        scores = avg_ratings
    return scores

# Step 3: Recommend
user_index = 0  # User1
predictions = predict_scores(user_index)
recommended_movie = np.argmax(predictions) + 1

print(f"\nUser{user_index+1}'s predicted scores: {predictions}")
print(f"Recommended movie for User{user_index+1}: Movie {recommended_movie}")
