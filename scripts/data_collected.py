# Script récupération des données OpenAgenda 10K events (valeurs manquantes gérées)
# data_collected.py
# scripts/data_collected.py
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

# Ignore le warning spécifique BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# === Configuration OpenAgenda ===
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
PARAMS = {
    "limit": 100,  # max par page
    "refine.location_countrycode": "FR",
    "where": "firstdate_begin >= '2025-01-01' AND lastdate_end <= '2026-01-01'",
    "order_by": "firstdate_begin ASC",
    "select": "uid,title_fr,description_fr,longdescription_fr,location_city,firstdate_begin,lastdate_end,location_coordinates,keywords_fr"
}

# === Fonctions ===
def fetch_events():
    """Récupère les événements avec pagination et limite API à 10 000."""
    print("⏳ Récupération des événements depuis OpenAgenda...")
    all_events = []
    offset = 0
    limit = PARAMS["limit"]

    while True:
        params = PARAMS.copy()
        params["offset"] = offset
        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print(f"❌ Erreur {response.status_code} lors de la requête.")
            print(response.text)
            break

        data = response.json()
        results = data.get("results", [])
        total_count = data.get("total_count", 0)

        if not results:
            break

        all_events.extend(results)
        offset += limit

        print(f"✅ {len(all_events)}/{total_count} événements récupérés...")

        # Limite API 10 000 événements
        if offset >= 10000:
            print("🚧 Limite API atteinte : arrêt à 10 000 événements.")
            break

        if offset >= total_count:
            break

    print(f"🎯 Total final : {len(all_events)} événements.")
    return all_events


def clean_html(text):
    """Supprime balises HTML et nettoie les espaces/sauts de ligne."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = " ".join(text.split())
    return text.strip()


def normalize_keywords(x):
    """Convertit les mots-clés en liste et gère les valeurs manquantes."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(k).strip() for k in x if str(k).strip()]
    if isinstance(x, str):
        return [k.strip() for k in x.split(",") if k.strip()]
    return []


def clean_events(events):
    """Nettoie et structure les événements."""
    df = pd.json_normalize(events)

    keep_cols = [
        "uid",
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "location_city",
        "firstdate_begin",
        "lastdate_end",
        "location_coordinates.lat",
        "location_coordinates.lon",
        "keywords_fr"
    ]
    df = df[[col for col in keep_cols if col in df.columns]]

    df.columns = [
        "id", "title", "description", "description_longue",
        "city", "start_date", "end_date", "latitude", "longitude", "keywords"
    ][:len(df.columns)]

    # Nettoyage des textes
    if "description" in df.columns:
        df["description"] = df["description"].apply(clean_html)
    if "description_longue" in df.columns:
        df["description_longue"] = df["description_longue"].apply(clean_html)

    # Remplacer les valeurs manquantes
    if "description_longue" in df.columns:
        df["description_longue"] = df["description_longue"].replace("", pd.NA)
        df["description_longue"] = df["description_longue"].fillna(df["description"])
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Ville inconnue")
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].fillna(0.0)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].fillna(0.0)

    # Normalisation des mots-clés
    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(normalize_keywords)

    # Supprimer les lignes essentielles manquantes
    df.dropna(subset=["id", "title", "description"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# === Script principal ===
def main():
    events = fetch_events()
    if not events:
        print("⚠️ Aucun événement récupéré.")
        return

    df = clean_events(events)
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    #output_path = f"data/processed/events_clean_{timestamp}.csv"  # Version avec horodatage
    output_path = f"data/processed/events_clean.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"💾 Données nettoyées et sauvegardées dans : {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()








