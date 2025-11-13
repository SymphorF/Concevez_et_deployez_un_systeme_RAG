# app_config.py
import os
from dotenv import load_dotenv

# Charge le fichier .env à la racine
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
if not OPENAGENDA_API_KEY:
    raise RuntimeError("OPENAGENDA_API_KEY non défini. Ajoute-le dans .env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY non défini. Ajoute-le dans .env")
