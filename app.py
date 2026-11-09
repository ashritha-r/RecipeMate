
import os
import pickle
import faiss
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import recipemate_clean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    with open(os.path.join(BASE_DIR, filename), "rb") as f:
        return pickle.load(f)

print("Loading RecipeMate data...")

recipemate_clean.recipes = load_pickle("processed_recipes.pkl")
recipemate_clean.recipe_ids = load_pickle("recipe_ids.pkl")
recipemate_clean.substitution_index = load_pickle("substitution_index.pkl")
recipemate_clean.index = faiss.read_index(
    os.path.join(BASE_DIR, "recipe_faiss.index")
)

print("Loading embedding model...")
recipemate_clean.model = SentenceTransformer("all-MiniLM-L6-v2")

print("RecipeMate is ready!")

app = Flask(__name__, template_folder=".")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"response": "Please enter a message."})

    try:
        response = recipemate_clean.chatbot_response(message)
        return jsonify({"response": response})
    except Exception as e:
        print("Chat error:", e)
        return jsonify({"response": "Something went wrong. Please try again."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
