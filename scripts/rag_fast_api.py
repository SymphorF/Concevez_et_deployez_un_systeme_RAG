
# Code avec uniquement le mode short déscription et filtrage par ville
# === rag_fast_api.py ===

import os
import faiss
import pickle
import time
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from mistralai.models import SDKError
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from app_config import MISTRAL_API_KEY
from fastapi import BackgroundTasks
from pydantic import BaseModel
from fastapi import Body

class AskRequest(BaseModel):
    question: str
    k: int = 5

# ==============================
# 1️⃣ Classe Mistral Embedding
# ==============================
class MistralEmbedding(Embeddings):
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def embed_query(self, text: str, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model="mistral-embed",
                    inputs=[text]
                )
                return response.data[0].embedding
            except SDKError as e:
                err = str(e)
                if any(err_part in err for err_part in ["rate_limited", "429", "service_tier_capacity_exceeded"]):
                    wait = 2 ** attempt + 2
                    print(f"⚠️ Rate limit, attente {wait}s (tentative {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise
        raise Exception("🚫 Trop de tentatives échouées pour l'embedding")

    def embed_documents(self, texts: list[str], max_retries=5):
        """Embeddings pour une liste de documents (obligatoire pour LangChain)"""
        embeddings = []
        for text in texts:
            emb = self.embed_query(text, max_retries=max_retries)
            embeddings.append(emb)
        return embeddings


# ==============================
# 2️⃣ Chargement de l’index FAISS
# ==============================

def load_faiss_index(index_name: str, embedding_function):
    INDEX_DIR = os.path.join("data", "processed", "faiss_indexes")
    index_path = os.path.join(INDEX_DIR, f"{index_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{index_name}_metadata.pkl")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        documents = pickle.load(f)

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    print(f"✅ Index '{index_name}' chargé ({len(documents)} documents)")

    return FAISS(embedding_function=embedding_function, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id), documents


# ==============================
# 3️⃣ Extraction et filtrage
# ==============================
def extract_unique_cities(documents):
    """Extrait les villes présentes dans les métadonnées."""
    cities = {doc.metadata.get("city", "").lower().strip() for doc in documents if doc.metadata.get("city")}
    return list(cities)


def parse_query(query: str, available_cities):
    """Détecte la ville et la période dans la requête."""
    q = query.lower()
    city = next((c for c in available_cities if c in q), None)
    now = datetime.now()

    if "ce week" in q:
        date_min, date_max = now, now + timedelta(days=7)
    elif "demain" in q:
        date_min, date_max = now + timedelta(days=1), now + timedelta(days=1)
    elif "aujourd'hui" in q:
        date_min, date_max = now, now
    elif "mois prochain" in q:
        date_min, date_max = now + timedelta(days=25), now + timedelta(days=55)
    else:
        date_min, date_max = None, None

    return city, date_min, date_max


def filter_results(results, city=None, date_min=None, date_max=None):
    """Filtre les résultats par ville et date."""
    filtered = []
    for doc, score in results:
        meta = doc.metadata
        keep = True

        if city and "city" in meta:
            if city.lower() not in str(meta["city"]).lower():
                keep = False

        if keep and date_min and "start_date" in meta:
            try:
                event_date = parse_date(meta["start_date"])
                if not (date_min <= event_date <= date_max):
                    keep = False
            except Exception:
                pass

        if keep:
            filtered.append((doc, score))
    return filtered


# ==============================
# 4️⃣ Configuration FastAPI
# ==============================
app = FastAPI(title="RAG Event Search API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Ajout dans la section du chargement de FAISS ===
print("🚀 Chargement des index FAISS (short et long)...")
embedding_function = MistralEmbedding(api_key=MISTRAL_API_KEY)

vs_short, all_docs_short = load_faiss_index("index_descriptions", embedding_function)
vs_long, all_docs_long = load_faiss_index("index_descriptions_longues", embedding_function)

all_docs = all_docs_short + all_docs_long
available_cities = extract_unique_cities(all_docs)
print(f"🏙️ {len(available_cities)} villes détectées.")


# ==============================
# 5️⃣ Routes principales
# ==============================


@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API RAG Event Search 🎭",
        "usage": "/search?query=votre_requête&k=nombre",
        "exemple": "/search?query=concert+jazz+à+Lyon&k=5"
    }

# === Modification de la route /search ===
@app.get("/search")
def search_events(
    query: str = Query(..., description="Requête (ex: 'concert jazz à Paris')"), 
    k: int = Query(5, ge=1, le=10)
):
    """ 
    Recherche sémantique dans les événements culturels.
    
    Exemples de requêtes:
    - "Spectacle danse Bordeaux"
    - "phtographe à Fougères"
    - "Concert de jazz à Lyon"
    - "exposition photo à Paris" 
    - "ateliers cuisine pour enfants"
    - "spectacle de danse à Bordeaux"
    """
    
    city, date_min, date_max = parse_query(query, available_cities)
    print(f"\n🔎 Query: {query} → Ville: {city}, Dates: {date_min} à {date_max}")

    try:
        results_short = vs_short.similarity_search_with_score(query, k=k)
        results_long = vs_long.similarity_search_with_score(query, k=k)

        # Fusion et tri des résultats
        all_results = results_short + results_long
        all_results.sort(key=lambda x: x[1])  # trier par score croissant (plus petit = plus pertinent)

        filtered = filter_results(all_results, city, date_min, date_max)

        formatted = [
            {
                "id": str(doc.metadata.get("id", "unknown")),
                "title": doc.metadata.get("title"),
                "city": doc.metadata.get("city"),
                "start_date": doc.metadata.get("start_date"),
                "end_date": doc.metadata.get("end_date"),
                "score": float(score),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc, score in filtered
        ]

        return {
            "query": query,
            "city_detected": city,
            "results_count": len(formatted),
            "results": formatted
        }

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {"error": str(e), "results": []}


@app.post("/ask")
def ask_question(request: AskRequest):
    """Pose une question au système RAG et renvoie une réponse générée."""
    query = request.question
    k = request.k
    city, date_min, date_max = parse_query(query, available_cities)

    # Recherche sémantique
    results_short = vs_short.similarity_search_with_score(query, k=k)
    results_long = vs_long.similarity_search_with_score(query, k=k)
    all_results = results_short + results_long
    all_results.sort(key=lambda x: x[1])

    filtered = filter_results(all_results, city, date_min, date_max)
    context = "\n\n".join([doc.page_content for doc, _ in filtered[:k]])

    # 🔹 Génération de réponse augmentée avec gestion d’erreurs et fallback
    response_text = None
    max_retries = 3
    wait_time = 2
    model = "mistral-large-latest"

    for attempt in range(max_retries):
        try:
            completion = embedding_function.client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant culturel. Donne une réponse claire, concise et utile."},
                    {"role": "user", "content": f"Question: {query}\n\nContexte:\n{context}"}
                ],
            )

            # ✅ Correction de la lecture du contenu
            response_text = completion.choices[0].message.content
            break

        except SDKError as e:
            err = str(e)
            if "429" in err or "capacity_exceeded" in err:
                print(f"⚠️ Saturation du modèle ({model}), tentative {attempt+1}/{max_retries} — attente {wait_time}s")
                time.sleep(wait_time)
                wait_time *= 2
                if attempt == max_retries - 1 and model == "mistral-large-latest":
                    print("⏬ Passage au modèle de secours : mistral-small-latest")
                    model = "mistral-small-latest"
                    attempt = 0
                    wait_time = 2
                    continue
            else:
                response_text = f"Erreur API : {err}"
                break
        except Exception as e:
            response_text = f"Erreur inattendue : {str(e)}"
            break

    # Si aucune réponse n’a pu être générée
    if not response_text:
        response_text = "Désolé, je n’ai pas pu générer de réponse pour le moment. Réessaie un peu plus tard."

    return {
        "question": query,
        "city_detected": city,
        "results_used": len(filtered),
        "generated_answer": response_text.strip()
    }


@app.post("/rebuild")
def rebuild_indexes(background_tasks: BackgroundTasks):
    """
    🔄 Reconstruit les index FAISS depuis la collecte des données OpenAgenda pour la mise à jour.
    """

    def rebuild_process():
        import pandas as pd
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        import numpy as np

        print("🚧 Reconstruction des index FAISS en cours...")

        # === 1️⃣ Chargement des données ===
        data_path = os.path.join("data", "processed", "events_clean_20251103_1212.csv")
        
        if not os.path.exists(data_path):
            print(f"❌ Fichier non trouvé: {data_path}")
            return
        
        df = pd.read_csv(data_path)
        print(f"📊 {len(df)} événements chargés")

        # ÉCHANTILLONNAGE : Prendre seulement 1000 événements pour les tests
        sample_size = min(1000, len(df))
        df = df.head(sample_size)
        print(f"🎯 Utilisation d'un échantillon de {len(df)} événements pour éviter les rate limits")

        # === 2️⃣ Création des Documents ===
        docs_short = [
            Document(
                page_content=str(row["description"]),
                metadata={
                    "id": row["id"],
                    "title": row.get("title", ""),
                    "city": row.get("city", ""),
                    "start_date": row.get("start_date", ""),
                    "end_date": row.get("end_date", "")
                }
            )
            for _, row in df.iterrows()
        ]

        docs_long = [
            Document(
                page_content=str(row["description_longue"]),
                metadata={
                    "id": row["id"],
                    "title": row.get("title", ""),
                    "city": row.get("city", ""),
                    "start_date": row.get("start_date", ""),
                    "end_date": row.get("end_date", "")
                }
            )
            for _, row in df.iterrows()
        ]

        print(f"📄 {len(docs_short)} documents créés")

        # === 3️⃣ Création des embeddings AVEC PAUSES ===
        embedding_function = MistralEmbedding(api_key=MISTRAL_API_KEY)

        print("🔄 Création des embeddings avec pauses...")
        
        # Créer le premier index
        print("📝 Index des descriptions courtes...")
        vs_short = FAISS.from_documents(docs_short, embedding_function)
        
        # Pause de 30 secondes entre les deux index
        print("⏳ Pause de 30 secondes pour éviter les rate limits...")
        time.sleep(30)
        
        print("📝 Index des descriptions longues...")
        vs_long = FAISS.from_documents(docs_long, embedding_function)

        # === 4️⃣ Sauvegarde des index ===
        INDEX_DIR = os.path.join("data", "processed", "faiss_indexes")
        os.makedirs(INDEX_DIR, exist_ok=True)

        vs_short.save_local(os.path.join(INDEX_DIR, "index_descriptions"))
        vs_long.save_local(os.path.join(INDEX_DIR, "index_descriptions_longues"))

        print("✅ Index FAISS reconstruits avec succès !")
        print(f"📁 Fichiers sauvegardés dans: {INDEX_DIR}")

    background_tasks.add_task(rebuild_process)

    return {
        "status": "success", 
        "message": "Reconstruction lancée avec échantillonnage (1000 événements)",
        "duration": "~5-10 minutes estimées"
    }

@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "indexes_loaded": True
    }

# ==============================
# 6️⃣ Lancement local
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_fast_api:app", host="0.0.0.0", port=8000, reload=True)

