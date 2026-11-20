# Asisten Praktis Pastoral & Liturgi Gereja Katolik (Hybrid RAG)

## Cara Menjalankan di Komputer Lokal

1. **Clone repository ini:**
   ```bash
   git clone <URL_REPOSITORY_ANDA>
   cd asisten-pastoral-rag
   ```

2. **Buat dan aktifkan virtual environment (Python 3.11):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Instal dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi API Key:**
   Duplikat file `.env.example` menjadi `.env`, lalu isi dengan `GEMINI_API_KEY` dari Google AI Studio.

5. **Bangun Database Vektor (Indexing):**
   ```bash
   python src/ingest.py
   ```

6. **Jalankan Aplikasi Web:**
   ```bash
   streamlit run app.py
   ```