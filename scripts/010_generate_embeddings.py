'''
# generate_embeddings.py
import os
import pandas as pd
import time
from tqdm import tqdm
from mistralai import Mistral
from config import MISTRAL_API_KEY

# === Configuration ===
INPUT_CSV = "data/processed/events_clean_20251031_1155.csv"  # Ton CSV nettoyé
OUTPUT_CSV = "data/processed/events_with_embeddings.csv"
BATCH_SAVE = 20  # Sauvegarde toutes les 20 lignes

# === Initialisation du client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

def embed_text(text, max_retries=5):
    """Génère l'embedding pour un texte via Mistral avec retry en cas de rate limit."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            err_str = str(e)
            if "rate_limited" in err_str or "Service tier capacity" in err_str:
                wait = 2 ** i  # backoff exponentiel
                print(f"Erreur rate limit ou capacité, attente {wait}s... retry {i+1}/{max_retries}")
                time.sleep(wait)
            else:
                print(f"Erreur embedding: {e}")
                return None
    print("Échec après plusieurs retries")
    return None

def main():
    # Chargement du CSV nettoyé
    df = pd.read_csv(INPUT_CSV)

    # Vérifier que la colonne 'description' existe
    if 'description' not in df.columns:
        raise RuntimeError("Colonne 'description' introuvable dans le CSV")

    # Initialiser ou récupérer les embeddings existants
    embeddings = df['embedding'].tolist() if 'embedding' in df.columns else []

    # Génération des embeddings
    print("Génération des embeddings...")
    for i, desc in enumerate(tqdm(df['description'])):
        if i < len(embeddings) and embeddings[i] is not None:
            continue  # Skip si déjà généré

        vec = embed_text(str(desc))
        if i < len(embeddings):
            embeddings[i] = vec
        else:
            embeddings.append(vec)

        # Sauvegarde périodique
        if (i + 1) % BATCH_SAVE == 0 or (i + 1) == len(df):
            df['embedding'] = embeddings
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"Sauvegarde intermédiaire à la ligne {i+1}")

    # Sauvegarde finale
    df['embedding'] = embeddings
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Embeddings ajoutés et sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
'''
















# Generation des embeddings déjà fait (sans chunking)
'''
import os
import pandas as pd
import time
from tqdm import tqdm
from mistralai import Mistral
from config import MISTRAL_API_KEY

# === Configuration ===
INPUT_CSV = "data/processed/events_clean_20251103_1212.csv"
OUTPUT_CSV = "data/processed/events_with_embeddings.csv"
BATCH_SAVE = 20

# === Initialisation du client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

def embed_text(text, max_retries=5):
    """Génère l'embedding pour un texte via Mistral avec retry en cas de rate limit."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
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
    print("🚫 Échec après plusieurs retries")
    return None

def main():
    df = pd.read_csv(INPUT_CSV)

    if 'description' not in df.columns:
        raise RuntimeError("Colonne 'description' introuvable dans le CSV")

    # Crée la colonne 'embedding' si absente
    if 'embedding' not in df.columns:
        df['embedding'] = [None] * len(df)

    print("🚀 Génération des embeddings...")
    for i in tqdm(range(len(df))):
        if pd.notna(df.loc[i, 'embedding']):
            continue  # déjà calculé

        desc = str(df.loc[i, 'description'])
        embedding = embed_text(desc)
        df.at[i, 'embedding'] = embedding

        # Sauvegarde intermédiaire
        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Sauvegarde intermédiaire à la ligne {i+1}")

    # Sauvegarde finale
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Embeddings ajoutés et sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
'''


















# Generation des embeddings avec chunking
'''
import os
import pandas as pd
import time
from tqdm import tqdm
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import MISTRAL_API_KEY

# === Configuration ===
INPUT_CSV = "data/processed/events_clean_20251103_1212.csv"
OUTPUT_CSV = "data/processed/events_with_embeddings.csv"
BATCH_SAVE = 20

# === Initialisation du client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

# === Initialisation du text splitter (chunking) ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # taille de chaque segment
    chunk_overlap=50,    # chevauchement entre les segments
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]  # découpage logique
)

def embed_text(text, max_retries=5):
    """Génère l'embedding pour un texte via Mistral avec retry en cas de rate limit."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
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
    print("🚫 Échec après plusieurs retries")
    return None

def main():
    df = pd.read_csv(INPUT_CSV)

    if 'description' not in df.columns:
        raise RuntimeError("Colonne 'description' introuvable dans le CSV")

    # Crée la colonne 'embedding' si absente
    if 'embedding' not in df.columns:
        df['embedding'] = [None] * len(df)

    print("🚀 Génération des embeddings avec chunking...")

    for i in tqdm(range(len(df))):
        if pd.notna(df.loc[i, 'embedding']):
            continue  # déjà calculé

        desc = str(df.loc[i, 'description'])
        if not desc or desc.lower() == "nan":
            df.at[i, 'embedding'] = None
            continue

        # --- CHUNKING du texte ---
        chunks = text_splitter.split_text(desc)

        # Génère un embedding pour chaque chunk
        chunk_embeddings = []
        for chunk in chunks:
            emb = embed_text(chunk)
            if emb:
                chunk_embeddings.append(emb)
            time.sleep(0.2)  # petite pause pour éviter le rate limit

        # Moyenne des embeddings des chunks pour représenter la description complète
        if chunk_embeddings:
            # moyenne sur chaque dimension du vecteur
            import numpy as np
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            df.at[i, 'embedding'] = avg_embedding
        else:
            df.at[i, 'embedding'] = None

        # Sauvegarde intermédiaire
        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Sauvegarde intermédiaire à la ligne {i+1}")

    # Sauvegarde finale
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Embeddings ajoutés et sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
'''
















# Generation des embeddings avec chunking et sleep

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
BATCH_SAVE = 20
EMBED_SLEEP = 0.2     # Délai entre chaque appel API (en secondes)
MAX_RETRIES = 5

# === Modes ===
MODE_TEST = True      # 🔁 Passe à False pour traiter TOUT le dataset
TEST_SIZE = 1000      # Nombre de lignes à traiter en mode test

# === Initialisation du client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

# === Initialisation du text splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

def embed_text(text, max_retries=MAX_RETRIES):
    """Génère l'embedding d'un texte via Mistral avec retries en cas de rate limit."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
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
    print("🚫 Échec après plusieurs retries")
    return None


def main():
    df = pd.read_csv(INPUT_CSV)

    if 'description' not in df.columns:
        raise RuntimeError("Colonne 'description' introuvable dans le CSV")

    # Mode test : limite le nombre de lignes
    if MODE_TEST:
        df = df.head(TEST_SIZE)
        print(f"🧪 Mode TEST activé : traitement des {len(df)} premières lignes")
    else:
        print(f"🚀 Mode COMPLET activé : traitement des {len(df)} lignes")

    # Crée la colonne 'embedding' si absente
    if 'embedding' not in df.columns:
        df['embedding'] = [None] * len(df)

    print("⚙️  Génération des embeddings avec chunking...\n")

    for i in tqdm(range(len(df))):
        if pd.notna(df.loc[i, 'embedding']):
            continue  # déjà calculé

        desc = str(df.loc[i, 'description'])
        if not desc or desc.lower() == "nan":
            df.at[i, 'embedding'] = None
            continue

        # --- CHUNKING du texte ---
        chunks = text_splitter.split_text(desc)

        # --- Génère un embedding pour chaque chunk ---
        chunk_embeddings = []
        for chunk in chunks:
            emb = embed_text(chunk)
            if emb:
                chunk_embeddings.append(emb)
            time.sleep(EMBED_SLEEP)  # Pause entre chaque requête

        # --- Moyenne des embeddings des chunks ---
        if chunk_embeddings:
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            df.at[i, 'embedding'] = avg_embedding
        else:
            df.at[i, 'embedding'] = None

        # --- Sauvegarde intermédiaire ---
        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Sauvegarde intermédiaire à la ligne {i+1}")

    # === Sauvegarde finale ===
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Embeddings ajoutés et sauvegardés dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
'''



















# Génération des embeddings avec chunking et sleep generation embeddings sur "description" et "description_longue" (Version finale)
# generate_embeddings.py

# generate_embeddings.py

import os
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import MISTRAL_API_KEY

# === Configuration ===
#INPUT_CSV = "data/processed/events_clean_20251103_1212.csv" # Le CSV nettoyé recupéré depuis data_collected.py
INPUT_CSV = "data/processed/events_clean.csv" # Le CSV nettoyé recupéré depuis data_collected.py
OUTPUT_CSV = "data/processed/events_with_embeddings.csv"
BATCH_SAVE = 20
EMBED_SLEEP = 0.2     # Délai entre chaque appel API (en secondes)
MAX_RETRIES = 5

# === Modes ===
MODE_TEST = False      # 🔁 Passe à True pour tester sur un échantillon
TEST_SIZE = 1000       # Nombre de lignes à traiter en mode test

# === Initialisation du client Mistral ===
client = Mistral(api_key=MISTRAL_API_KEY)

# === Initialisation du text splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)


def embed_text(text, max_retries=MAX_RETRIES):
    """Génère l'embedding d'un texte via Mistral avec retries en cas de rate limit."""
    for i in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
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
    print(" Échec après plusieurs retries")
    return None

# Fonction principale de génération des embeddings pour une colonne donnée
def generate_embedding_for_column(df, column_name, new_column_name):
    """Génère les embeddings pour une colonne spécifique du DataFrame."""
    print(f"\n  Génération des embeddings pour '{column_name}'...\n")

    # Crée la colonne d'embeddings si absente
    if new_column_name not in df.columns:
        df[new_column_name] = [None] * len(df)

    for i in tqdm(range(len(df))):
        if pd.notna(df.loc[i, new_column_name]):
            continue  # déjà calculé

        text = str(df.loc[i, column_name])
        if not text or text.lower() == "nan":
            df.at[i, new_column_name] = None
            continue

        # --- CHUNKING du texte ---
        chunks = text_splitter.split_text(text)

        # --- Génère un embedding pour chaque chunk ---
        chunk_embeddings = []
        for chunk in chunks:
            emb = embed_text(chunk)
            if emb:
                chunk_embeddings.append(emb)
            time.sleep(EMBED_SLEEP)  # Pause entre chaque requête

        # --- Moyenne des embeddings des chunks ---
        if chunk_embeddings:
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            df.at[i, new_column_name] = avg_embedding
        else:
            df.at[i, new_column_name] = None

        # --- Sauvegarde intermédiaire ---
        if (i + 1) % BATCH_SAVE == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f" Sauvegarde intermédiaire à la ligne {i+1} ({column_name})")

    return df

# === Main === 
def main():
    df = pd.read_csv(INPUT_CSV)

    # Vérification des colonnes nécessaires
    if 'description' not in df.columns:
        raise RuntimeError("Colonne 'description' introuvable dans le CSV")
    if 'description_longue' not in df.columns:
        raise RuntimeError("Colonne 'description_longue' introuvable dans le CSV")

    # Mode test : limite le nombre de lignes
    if MODE_TEST:
        df = df.head(TEST_SIZE)
        print(f" Mode TEST activé : traitement des {len(df)} premières lignes")
    else:
        print(f" Mode COMPLET activé : traitement de {len(df)} lignes")

    # Génération pour les deux colonnes
    df = generate_embedding_for_column(df, "description", "embedding_description")
    df = generate_embedding_for_column(df, "description_longue", "embedding_description_longue")

    # Sauvegarde finale
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Embeddings ajoutés et sauvegardés dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
