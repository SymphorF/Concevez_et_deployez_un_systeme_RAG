# resume_generate_embeddings.py
'''
import os
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import MISTRAL_API_KEY

# === Configuration ===
INPUT_CSV = "data/processed/events_clean_20251103_1212.csv"
OUTPUT_CSV = "data/processed/events_with_embeddings.csv"
TEMP_SAVE = "data/processed/events_with_embeddings_temp.csv"
BATCH_SAVE = 20
EMBED_SLEEP = 0.2
MAX_RETRIES = 5

# === Modes ===
MODE_TEST = False
TEST_SIZE = 1000

# === Client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

# === Text Splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

def embed_text(text, max_retries=MAX_RETRIES):
    for i in range(max_retries):
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=[text])
            return response.data[0].embedding
        except Exception as e:
            err_str = str(e)
            if "rate_limited" in err_str or "Service tier capacity" in err_str:
                wait = 2 ** i
                print(f"⚠️ Rate limit ou capacité, attente {wait}s... (retry {i+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"❌ Erreur embedding : {e}")
                return None
    print("❌ Échec après plusieurs retries")
    return None

def generate_embedding_for_column(df, column_name, new_column_name):
    print(f"\n⚙️ Génération des embeddings pour '{column_name}'...\n")

    if new_column_name not in df.columns:
        df[new_column_name] = [None] * len(df)

    for i in tqdm(range(len(df))):
        # ⏩ Skip si déjà calculé
        if pd.notna(df.loc[i, new_column_name]) and isinstance(df.loc[i, new_column_name], str) and df.loc[i, new_column_name].startswith('['):
            continue

        text = str(df.loc[i, column_name])
        if not text or text.lower() == "nan":
            df.at[i, new_column_name] = None
            continue

        # Découpe du texte
        chunks = text_splitter.split_text(text)
        chunk_embeddings = []

        for chunk in chunks:
            emb = embed_text(chunk)
            if emb:
                chunk_embeddings.append(emb)
            time.sleep(EMBED_SLEEP)

        # Moyenne des chunks
        if chunk_embeddings:
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            df.at[i, new_column_name] = avg_embedding
        else:
            df.at[i, new_column_name] = None

        # Sauvegarde périodique
        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(TEMP_SAVE, index=False)
            print(f"💾 Sauvegarde temporaire à la ligne {i+1}")

    return df

def main():
    # 🔁 Reprise automatique
    if os.path.exists(OUTPUT_CSV):
        print(f"🔄 Reprise depuis {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    elif os.path.exists(TEMP_SAVE):
        print(f"🔄 Reprise depuis {TEMP_SAVE}")
        df = pd.read_csv(TEMP_SAVE)
    else:
        print(f"🆕 Démarrage depuis {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)

    if MODE_TEST:
        df = df.head(TEST_SIZE)
        print(f" Mode TEST activé : {len(df)} lignes")

    # Embeddings pour les deux colonnes
    df = generate_embedding_for_column(df, "description", "embedding_description")
    df = generate_embedding_for_column(df, "description_longue", "embedding_description_longue")

    # Sauvegarde finale
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Embeddings complétés et sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
'''



















# resume_generate_embeddings_fixed.py
'''
import os
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import MISTRAL_API_KEY

# === Configuration ===
INPUT_CSV = "data/processed/events_clean_20251103_1212.csv"
OUTPUT_CSV = "data/processed/events_with_embeddings.csv"
TEMP_SAVE = "data/processed/events_with_embeddings_temp.csv"
BATCH_SAVE = 20
EMBED_SLEEP = 0.2
MAX_RETRIES = 5

# === Modes ===
MODE_TEST = False
TEST_SIZE = 1000

# === Client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

# === Text Splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

def embed_text(text, max_retries=MAX_RETRIES):
    """Appel API Mistral avec gestion du rate limit."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=[text])
            return response.data[0].embedding
        except Exception as e:
            err_str = str(e)
            if "rate_limited" in err_str or "Service tier capacity" in err_str:
                wait = 2 ** i
                print(f"⚠️ Rate limit ou capacité, attente {wait}s... (retry {i+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"❌ Erreur embedding : {e}")
                return None
    print("❌ Échec après plusieurs retries")
    return None

def generate_embedding_for_column(df, column_name, new_column_name):
    """Génère les embeddings pour une colonne donnée."""
    print(f"\n⚙️ Génération des embeddings pour '{column_name}'...")

    if new_column_name not in df.columns:
        df[new_column_name] = [None] * len(df)

    for i in tqdm(range(len(df))):
        val = df.loc[i, new_column_name]
        # Skip si déjà calculé (embedding déjà présent)
        if isinstance(val, str) and val.startswith('['):
            continue

        text = str(df.loc[i, column_name])
        if not text or text.lower() == "nan":
            df.at[i, new_column_name] = None
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
            df.at[i, new_column_name] = avg_embedding
        else:
            df.at[i, new_column_name] = None

        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(TEMP_SAVE, index=False)
            print(f"💾 Sauvegarde temporaire ({column_name}) à la ligne {i+1}")

    print(f"✅ Colonne '{column_name}' terminée.\nSauvegarde finale...")
    df.to_csv(OUTPUT_CSV, index=False)
    return df

def main():
    # 🔁 Chargement intelligent
    if os.path.exists(OUTPUT_CSV):
        print(f"🔄 Reprise depuis {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    elif os.path.exists(TEMP_SAVE):
        print(f"🔄 Reprise depuis {TEMP_SAVE}")
        df = pd.read_csv(TEMP_SAVE)
    else:
        print(f"🆕 Démarrage depuis {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)

    if MODE_TEST:
        df = df.head(TEST_SIZE)
        print(f" Mode TEST activé : {len(df)} lignes")

    # Génère d'abord les embeddings courtes
    if "embedding_description" not in df.columns or df["embedding_description"].isna().any():
        df = generate_embedding_for_column(df, "description", "embedding_description")

    # Puis les longues, uniquement si incomplètes
    if "embedding_description_longue" not in df.columns or df["embedding_description_longue"].isna().any():
        df = generate_embedding_for_column(df, "description_longue", "embedding_description_longue")

    print(f"\n✅ Toutes les embeddings sont complétées et sauvegardées dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
'''









import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import MISTRAL_API_KEY

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
