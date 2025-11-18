import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app_config import MISTRAL_API_KEY

# === Configuration ===
INPUT_CSV = "data/processed/events_with_embeddings.csv"  # Ton fichier existant
OUTPUT_CSV = "data/processed/events_with_embeddings_fixed.csv"
TARGET_COLUMN = "description_longue"  # ou "description_longue"
EMBED_COLUMN = "embedding_description_longue"  # adapte selon ton fichier
BATCH_SAVE = 5
EMBED_SLEEP = 0.2
MAX_RETRIES = 5

# === Client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

# === Text Splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

# === Fonction utilitaire ===
def embed_text(text, max_retries=MAX_RETRIES):
    """Génère un embedding pour un texte donné, avec retries."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=[text])
            return response.data[0].embedding
        except Exception as e:
            err_str = str(e)
            if "rate_limited" in err_str or "Service tier capacity" in err_str:
                wait = 2 ** i
                print(f"⚠️ Rate limit, attente {wait}s... (retry {i+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"❌ Erreur embedding : {e}")
                return None
    print("❌ Échec après plusieurs tentatives")
    return None

# === Programme principal ===
def main():
    print(f"📂 Chargement de {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Identifier les lignes manquantes
    missing_mask = df[EMBED_COLUMN].isna()
    missing_df = df[missing_mask].copy()

    print(f"🔎 {missing_df.shape[0]} lignes avec embeddings manquants à régénérer.\n")

    for i in tqdm(missing_df.index, desc="Régénération des embeddings manquants"):
        text = str(df.loc[i, TARGET_COLUMN])
        if not text or text.lower() == "nan":
            df.at[i, EMBED_COLUMN] = None
            continue

        chunks = text_splitter.split_text(text)
        chunk_embeddings = []

        for chunk in chunks:
            emb = embed_text(chunk)
            if emb:
                chunk_embeddings.append(emb)
            time.sleep(EMBED_SLEEP)

        if chunk_embeddings:
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            df.at[i, EMBED_COLUMN] = avg_embedding
        else:
            df.at[i, EMBED_COLUMN] = None

        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Sauvegarde temporaire à la ligne {i+1}")

    # Sauvegarde finale
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Fichier final sauvegardé : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
