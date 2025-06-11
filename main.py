from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for web app

class BookRecommendationAPI:
    def __init__(self, model_dir='book_recommendation_model'):
        self.model_dir = model_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.books_data = None
        self.model_info = None
        self.is_loaded = False
        
    def load_model(self):
        """Load pre-trained model"""
        try:
            print(f"Loading model from {self.model_dir}...")
            
            # Load TF-IDF vectorizer
            with open(f'{self.model_dir}/tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load TF-IDF matrix
            with open(f'{self.model_dir}/tfidf_matrix.pkl', 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            
            # Load books data
            with open(f'{self.model_dir}/books_data.json', 'r', encoding='utf-8') as f:
                self.books_data = json.load(f)
            
            # Load model info
            with open(f'{self.model_dir}/model_info.json', 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            
            self.is_loaded = True
            print(f"✅ Model loaded successfully!")
            print(f"📊 Books: {len(self.books_data)}")
            print(f"📝 Vocabulary size: {self.model_info['vocab_size']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess text (same as training)"""
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def get_recommendations(self, query, top_k=10, min_similarity=0.01):
        """Get book recommendations"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            if not processed_query:
                return {"error": "Invalid query"}
            
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top recommendations
            top_indices = similarities.argsort()[-top_k*2:][::-1]  # Get more to filter
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] >= min_similarity:
                    recommendations.append({
                        'book': self.books_data[idx],
                        'similarity': float(similarities[idx]),
                        'similarity_percent': round(float(similarities[idx]) * 100, 2)
                    })
            
            # Limit to top_k
            recommendations = recommendations[:top_k]
            
            return {
                "success": True,
                "query": query,
                "total_results": len(recommendations),
                "recommendations": recommendations,
                "model_info": self.model_info
            }
            
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}

# Initialize API
recommender = BookRecommendationAPI()

@app.route('/', methods=['GET', 'POST'])
def home():
   return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if recommender.is_loaded else "loading",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": recommender.is_loaded
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if not recommender.is_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_info": recommender.model_info,
        "total_books": len(recommender.books_data),
        "status": "ready"
    })

@app.route('/search', methods=['POST'])
def search_post():
    """Search recommendations via POST"""
    if not recommender.is_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 10)
    min_similarity = data.get('min_similarity', 0.01)
    
    result = recommender.get_recommendations(query, top_k, min_similarity)
    
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify(result)

@app.route('/search/<query>')
def search_get(query):
    """Search recommendations via GET"""
    if not recommender.is_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    
    top_k = request.args.get('top_k', 10, type=int)
    min_similarity = request.args.get('min_similarity', 0.01, type=float)
    
    result = recommender.get_recommendations(query, top_k, min_similarity)
    
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify(result)

@app.route('/books')
def get_books():
    """Get all books (paginated)"""
    if not recommender.is_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    books_page = recommender.books_data[start_idx:end_idx]
    
    return jsonify({
        "books": books_page,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": len(recommender.books_data),
            "pages": (len(recommender.books_data) + per_page - 1) // per_page
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Load model on startup
    print("🚀 Starting Book Recommendation API...")
    
    if not os.path.exists('book_recommendation_model'):
        print("❌ Model directory 'book_recommendation_model' not found!")
        print("💡 Please run the training script first to generate the model.")
        exit(1)
    
    # Load model
    if recommender.load_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting Flask server...")
        
        # Run server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    else:
        print("❌ Failed to load model. Exiting...")
        exit(1)