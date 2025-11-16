# Avec détection ville automatique

# rag_langchain_faiss.py avec détection automatique des villes
# === rag_langchain_faiss.py ===

import os
import faiss
import pickle
import time
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from mistralai import Mistral
from mistralai.models import SDKError
from app_config import MISTRAL_API_KEY


# ================================
# 1️⃣ CLASSE PERSONNALISÉE : Mistral Embedding AVEC RETRY
# ================================
class MistralEmbedding(Embeddings):
    """Classe d'embeddings avec gestion robuste des rate limits."""

    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def embed_documents(self, texts, max_retries=5):
        """Embed une liste de textes avec gestion des rate limits."""
        batch_size = 32
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model="mistral-embed",
                        inputs=batch
                    )
                    embeddings.extend([d.embedding for d in response.data])
                    print(f"✅ Batch {i // batch_size + 1} traité ({len(batch)} textes)")
                    break
                except SDKError as e:
                    if any(err in str(e) for err in ["service_tier_capacity_exceeded", "rate_limited", "429"]):
                        wait_time = 2 ** attempt + 5
                        print(f"⚠️ Rate limit batch, attente {wait_time}s... (tentative {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise e
        return embeddings

    def embed_query(self, text, max_retries=5):
        """Embed une requête avec retry automatique."""
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model="mistral-embed",
                    inputs=[text]
                )
                return response.data[0].embedding
            except SDKError as e:
                if any(err in str(e) for err in ["service_tier_capacity_exceeded", "rate_limited", "429"]):
                    wait_time = 2 ** attempt + 2
                    print(f"⚠️ Rate limit query, attente {wait_time}s... (tentative {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception("🚫 Échec query après plusieurs tentatives")


# ================================
# 2️⃣ CONFIGURATION
# ================================
INDEX_DIR = os.path.join("data", "processed", "faiss_indexes")
embedding_function = MistralEmbedding(api_key=MISTRAL_API_KEY)


# ================================
# 3️⃣ CHARGEMENT DES INDEX
# ================================
def load_faiss_index(index_name: str, embedding_function):
    index_path = os.path.join(INDEX_DIR, f"{index_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{index_name}_metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Fichiers manquants pour {index_name}")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        documents = pickle.load(f)

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    vectorstore = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    print(f"✅ Index '{index_name}' chargé avec {len(documents)} documents.")
    return vectorstore


# ================================
# 4️⃣ EXTRACTION AUTOMATIQUE DES VILLES
# ================================
def extract_unique_cities(vectorstores):
    """Extrait automatiquement toutes les villes contenues dans les métadonnées FAISS."""
    cities = set()
    for vs in vectorstores:
        for _, doc in vs.docstore._dict.items():
            city = doc.metadata.get("city")
            if city:
                cities.add(city.lower().strip())
    print(f"🏙️ {len(cities)} villes détectées automatiquement : {list(cities)[:10]} ...")
    return list(cities)


# ================================
# 5️⃣ FONCTIONS DE FILTRAGE INTELLIGENT
# ================================
def parse_query(query: str, available_cities):
    """Extrait automatiquement la ville et la temporalité à partir de la requête utilisateur."""
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
    """Filtre les résultats par ville et période."""
    filtered = []

    for doc in results:
        meta = doc.metadata
        keep = True

        # Filtrage ville
        if city and "city" in meta:
            if city.lower() not in str(meta["city"]).lower():
                keep = False

        # Filtrage date
        if keep and date_min and "start_date" in meta:
            try:
                event_date = parse_date(meta["start_date"])
                if not (date_min <= event_date <= date_max):
                    keep = False
            except Exception:
                pass

        if keep:
            filtered.append(doc)

    return filtered


# ================================
# 6️⃣ TEST DU SYSTÈME
# ================================
if __name__ == "__main__":
    try:
        # 🔹 Chargement des index FAISS
        vs_short = load_faiss_index("index_descriptions", embedding_function)
        vs_long = load_faiss_index("index_descriptions_longues", embedding_function)

        # 🔹 Détection automatique des villes présentes dans les métadonnées
        available_cities = extract_unique_cities([vs_short, vs_long])

        # 🔹 Exemple de requête

        #query = "Spectacle danse Bordeaux"
        #query = "phtographe à Fougères"
        query = "Concert de jazz à Lyon"
        #query = "Concert de jazz"
        

        #query = "photographe à Fougères"
        #query = "Tous les événements à Paris aujourd'hui"
        #query = "Ateliers cuisine"
        #query = "Photos"
        #query = "Séance photos à Nogent"
        #query = "Concert de jazz"
        #query = "Concert de jazz à Lyon"
        #query = "Concert de jazz à Lyon ce weekend"
        #query = "Concert de jazz à Paris ce weekend"
        #query = "Atelier à La Grand-Combe"
        #query = "Atelier cuisine pour enfant à La Grand-Combe"
        #query = "Atelier de cuisine"
        #query = "exposition photo à paris"
        #query = "Fête de la musique"
        #query = "Ateliers cuisine pour adultes à Marseille"
        #query = "Exposition photo à Paris en juin 2024"
        #query = "Spectacle de danse à Bordeaux le mois prochain"
        #query = "Événements culturels à Lille il y'a une semaine"
        #query = "Salon d'exposition à Nantes"
        #query = "Concert de rock à Nice demain"
        #query = "Festival de cinéma à Toulouse ce week-end"
        #query = "Atelier peinture pour enfants à Rennes aujourd'hui"
        #query = "Conférence sur l'art contemporain à Marseille le mois prochain"
        #query = "Marché artisanal à Strasbourg ce week-end"
        city, date_min, date_max = parse_query(query, available_cities)

        print(f"\n🔎 Requête : {query}")
        print(f"➡️ Filtres appliqués → Ville: {city}, Dates: {date_min} → {date_max}")

        # 🔹 Recherche et filtrage
        print("\n📂 Recherche FAISS (descriptions courtes)...")
        results_short = vs_short.similarity_search(query, k=10)
        filtered_short = filter_results(results_short, city, date_min, date_max)

        print(f"\n🎯 {len(filtered_short)} résultats pertinents après filtrage :")
        for i, doc in enumerate(filtered_short[:3], 1):
            print(f"\n🧩 Résultat {i}:")
            print(f"📖 {doc.page_content[:200]}...")
            print(f"📍 Métadonnées : {doc.metadata}")

    except Exception as e:
        print(f"❌ Erreur générale: {e}")
