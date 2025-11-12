# === faiss_index_with_metadata_existing_embeddings.py ===

import os
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
import pickle

# --- Chargement du CSV ---
csv_path = os.path.join("data", "processed", "events_with_embeddings.csv")
df = pd.read_csv(csv_path)

# --- Fonction de conversion string → liste ---
def str_to_list(s):
    try:
        return np.array(eval(s), dtype=np.float32)
    except Exception:
        return None

# --- Dossier de sortie ---
output_dir = os.path.join("data", "processed", "faiss_indexes")
os.makedirs(output_dir, exist_ok=True)

# --- Fonction de création d’un index FAISS avec métadonnées existantes ---
def build_faiss_index(df, text_col, embedding_col, index_name):
    print(f"\n=== Création de l'index FAISS pour : {index_name} ===")

    df_valid = df.dropna(subset=[text_col, embedding_col]).copy()
    embeddings = np.vstack(df_valid[embedding_col].apply(str_to_list).dropna())

    # Création des documents LangChain
    documents = []
    for _, row in df_valid.iterrows():
        metadata = {
            "id": row.get("id"),
            "title": row.get("title"),
            "city": row.get("city"),
            "start_date": row.get("start_date"),
            "end_date": row.get("end_date"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "keywords": row.get("keywords"),
        }
        documents.append(Document(page_content=row[text_col], metadata=metadata))

    # Création du véritable index FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ {index.ntotal} vecteurs indexés dans {index_name}")

    # Sauvegarde du fichier FAISS
    faiss_path = os.path.join(output_dir, f"{index_name}.index")
    faiss.write_index(index, faiss_path)

    # Sauvegarde séparée des métadonnées + documents
    metadata_path = os.path.join(output_dir, f"{index_name}_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(documents, f)

    print(f"💾 Index FAISS sauvegardé dans : {faiss_path}")
    print(f"💾 Métadonnées sauvegardées dans : {metadata_path}")

# --- Création des deux index ---
build_faiss_index(df, "description", "embedding_description", "index_descriptions")
build_faiss_index(df, "description_longue", "embedding_description_longue", "index_descriptions_longues")














