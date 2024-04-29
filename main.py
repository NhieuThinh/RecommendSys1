# import libraries
import pandas as pd
from flask import Flask, jsonify
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load the saved model
# with open("C:\\Users\\INTEL\\Downloads\\RecommendSys\\trained\\als_model.pkl", "rb") as f:
with open("als_model1.pkl", "rb") as f:
# with open("/home/thinhnguyen2104/mysite/trained/als_model.pkl", "rb") as f:
    model, user_mapping, product_mapping = pickle.load(f)


# Function to load the DataFrame
def load_data():
    # Load the data from the Excel file
    df = pd.read_excel("rating.xlsx")

    # Mapping usernames and product IDs to integer indices
    user_mapping = {user: i for i, user in enumerate(df['username'].unique())}
    product_mapping = {pid: i for i, pid in enumerate(df['pid'].unique())}

    # Apply mapping to DataFrame
    df['user_id'] = df['username'].map(user_mapping)
    df['product_id'] = df['pid'].map(product_mapping)

    return df, user_mapping


# Function to get top N recommendations for a given username
def get_recommendations(username, N=10):
    df, user_mapping = load_data()

    user_id = user_mapping.get(username)
    if user_id is None:  # Handle cold start case
        # Find top N products with the highest average rating
        top_n_average_ratings = df.groupby('pid')['rating'].mean().nlargest(N).index.tolist()
        return top_n_average_ratings

    # Get the user's vector
    user_vector = model.user_factors[user_id]

    # Get item factors
    item_factors = model.item_factors

    # Predict ratings for all items
    all_items_ratings = item_factors.dot(user_vector)

    # Filter out items already rated by the user
    rated_items = set(df[df['username'] == username]['pid'])
    unrated_items = [(pid, rating) for pid, rating in enumerate(all_items_ratings) if
                     pid not in rated_items and pid != 0]

    # Sort the unrated items by predicted rating
    unrated_items.sort(key=lambda x: x[1], reverse=True)

    # Recommend top N items
    top_n_recommendations = [pid for pid, _ in unrated_items[:N]]

    return top_n_recommendations


@app.route('/recommendation/<string:username>', methods=['GET'])
def calculate_similarity(username):
    recommendations = get_recommendations(username)
    print(recommendations)
    return jsonify(recommendations)



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
