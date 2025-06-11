import pandas as pd
import numpy as np
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

class BookRecommendationModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True
        )
        self.tfidf_matrix = None
        self.books_data = None
        self.model_info = {
            'created_at': None,
            'num_books': 0,
            'vocab_size': 0,
            'method': 'TF-IDF + Cosine Similarity'
        }
    
    def preprocess_text(self, text):
        """Preprocessing teks untuk bahasa Indonesia dan Inggris"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def combine_features(self, row):
        """Menggabungkan semua fitur teks menjadi satu"""
        features = []
        
        # Bobot untuk setiap kolom (title lebih penting)
        title = self.preprocess_text(row.get('Judul_Buku', ''))
        author = self.preprocess_text(row.get('Author', ''))
        description = self.preprocess_text(row.get('Deskripsi', ''))
        publisher = self.preprocess_text(row.get('Penerbit', ''))
        
        # Berikan bobot lebih pada judul dan deskripsi
        combined = f"{title} {title} {description} {author} {publisher}"
        
        return combined
    
    def train(self, excel_file_path):
        """Training model dari file Excel"""
        print("📚 Memulai training model rekomendasi buku...")
        
        # Load data
        print("📖 Membaca file Excel...")
        try:
            df = pd.read_excel(excel_file_path)
            print(f"✅ Berhasil membaca {len(df)} buku")
        except Exception as e:
            print(f"❌ Error membaca file: {e}")
            return False
        
        # Validasi kolom
        required_columns = ['Author', 'Judul_Buku', 'Deskripsi', 'Penerbit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Kolom yang hilang: {missing_columns}")
            return False
        
        # Preprocess data
        print("🔄 Preprocessing data...")
        df['combined_features'] = df.apply(self.combine_features, axis=1)
        
        # Remove empty rows
        df = df[df['combined_features'].str.strip() != '']
        
        # Fit TF-IDF
        print("🧠 Training TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(df['combined_features'])
        
        # Store data
        self.books_data = df.to_dict('records')
        
        # Update model info
        self.model_info.update({
            'created_at': datetime.now().isoformat(),
            'num_books': len(df),
            'vocab_size': len(self.vectorizer.vocabulary_),
            'method': 'TF-IDF + Cosine Similarity'
        })
        
        print(f"✅ Training selesai!")
        print(f"📊 Jumlah buku: {self.model_info['num_books']}")
        print(f"📝 Ukuran vocabulary: {self.model_info['vocab_size']}")
        
        return True
    
    def save_model(self, model_dir='model'):
        """Simpan model dan data ke file"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        print("💾 Menyimpan model...")
        
        # Save TF-IDF vectorizer
        with open(f'{model_dir}/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save TF-IDF matrix
        with open(f'{model_dir}/tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        # Save books data
        with open(f'{model_dir}/books_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.books_data, f, ensure_ascii=False, indent=2)
        
        # Save model info
        with open(f'{model_dir}/model_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Model disimpan di folder: {model_dir}")
        print("📁 File yang dibuat:")
        print("   - tfidf_vectorizer.pkl")
        print("   - tfidf_matrix.pkl") 
        print("   - books_data.json")
        print("   - model_info.json")
    
    def load_model(self, model_dir='model'):
        """Load model dari file"""
        print("📂 Loading model...")
        
        # Load TF-IDF vectorizer
        with open(f'{model_dir}/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load TF-IDF matrix
        with open(f'{model_dir}/tfidf_matrix.pkl', 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        # Load books data
        with open(f'{model_dir}/books_data.json', 'r', encoding='utf-8') as f:
            self.books_data = json.load(f)
        
        # Load model info
        with open(f'{model_dir}/model_info.json', 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)
        
        print("✅ Model berhasil di-load!")
        print(f"📊 Jumlah buku: {self.model_info['num_books']}")
        print(f"📝 Ukuran vocabulary: {self.model_info['vocab_size']}")
        print(f"📅 Dibuat pada: {self.model_info['created_at']}")
        
        return True
    
    def get_recommendations(self, query, top_k=10):
        """Mendapatkan rekomendasi berdasarkan query"""
        if self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                recommendations.append({
                    'book': self.books_data[idx],
                    'similarity': float(similarities[idx]),
                    'similarity_percent': round(similarities[idx] * 100, 2)
                })
        
        return recommendations

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi model
    model = BookRecommendationModel()
    
    # Path ke file Excel Anda
    excel_file = "databuku.xlsx"  # Ganti dengan path file Anda
    
    # Training
    if model.train(excel_file):
        # Simpan model
        model.save_model("book_recommendation_model")
        
        # Test model (opsional)
        print("\n🔍 Testing model...")
        recommendations = model.get_recommendations("statistika", top_k=5)
        
        if recommendations:
            print(f"Top 5 rekomendasi untuk 'statistika':")
            for i, rec in enumerate(recommendations, 1):
                book = rec['book']
                print(f"{i}. {book['Judul_Buku']} - {book['Author']} ({rec['similarity_percent']:.1f}%)")
        else:
            print("Tidak ada rekomendasi ditemukan.")
    
    print("\n🎉 Proses selesai!")
    print("📋 Langkah selanjutnya:")
    print("1. Copy folder 'book_recommendation_model' ke direktori web")
    print("2. Gunakan web app untuk inference")