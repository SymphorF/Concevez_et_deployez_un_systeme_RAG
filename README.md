# 🤖 Assistant intelligent de recommandation d’événements culturels (Système RAG)

## 🎯 Objectif du projet

***Ce projet a pour but de concevoir et déployer un assistant intelligent capable de recommander des événements culturels à partir d’une base de données.***

**Le système repose sur une architecture RAG (Retrieval-Augmented Generation) combinant FAISS, LangChain, Mistral Embeddings, et FastAPI.**

***L’objectif : permettre la recherche sémantique et la génération de réponses contextualisées et pertinentes.***


## 📁 Structure du projet

``` 
Projet-8/
├── .github/workflows/                       # Fichiers YAML définissant les workflows CI/CD pour GitHub Actions
├── documents/                               # Documents liés au projet (captures d’écran, présentations, rapports, etc.)
├── notebooks/                               # Notebooks Jupyter pour l’exploration, l’analyse et les tests expérimentaux
├── scripts/                                 # Ensemble des scripts nécessaires à la collecte, au traitement et à l’indexation des données
├── tests/                                   # Tests unitaires et fonctionnels pour vérifier la fiabilité du code
│
├── .gitignore                               # Fichiers et dossiers à exclure du suivi Git
├── Dockerfile                               # Configuration Docker pour créer l’image du projet
├── docker-compose.yml                       # Orchestration multi-conteneurs (API, base de données, monitoring, etc.)
│
├── poetry.lock                              # Fichier de verrouillage des dépendances (généré automatiquement par Poetry)
├── pyproject.toml                           # Fichier principal de configuration du projet et de ses dépendances (Poetry)
├── requirements.txt                         # Liste des dépendances minimales pour exécuter le projet sans Poetry
├── lien.txt                                 # Liens utiles ou références vers le dépôt GitHub et autres ressources
└── README.md                                # Documentation principale du projet (ce fichier)
```


### ⚙️ Installation de l’environnement virtuel
### 🧩 1. Création de l’environnement virtuel

***Ouvrez le dossier de projet dans VS Code, puis lance le terminal intégré :***

  - Raccourci : Ctrl + ù (ou Ctrl + J)
  - Menu : Affichage > Terminal
  - Ensuite, exécute :
      python -m venv .venv

***👉 Cela crée un dossier .venv (ou le nom de ton choix) contenant l’environnement virtuel.***

### ⚡ 2. Activation de l’environnement virtuel

***Sous Windows :***

.venv\Scripts\activate

***Sous macOS / Linux :***

source .venv/bin/activate

### 🔍 3. Vérification de l’environnement actif

    - where python      # Windows
    - which python      # macOS / Linux

Vous devrez voir un chemin pointant vers le dossier .venv.

### 📦 4. Installation des dépendances de base

pip install numpy pandas fastapi

### 🧪 5. Vérification sur une installation propre

Vous pouvez simuler une installation propre avec :

    - poetry env remove python
    - poetry install
    - poetry run python test_imports.py

## 🚀 Étapes du projet

    Avant toute manipulation, assure-toi d’être dans ton environnement virtuel et à la racine du projet.

    C:\Users\...\Concevez_et_deployez_un_systeme_RAG>

    poetry shell       # Active ton environnement virtuel

    cd scripts


### 1️⃣ Collecte des données

    Collecte les données brutes depuis les sources externes :

    python 000_data_collected.py

### 2️⃣ Génération des embeddings

    Génère les embeddings sur les colonnes de description :

    python 010_generate_embeddings.py


***💡 Astuce :***

    Pour tester sur un échantillon limité, mets MODE_TEST = True

    Défini TEST_SIZE pour le nombre de lignes à traiter

    En cas d’arrêt ou d’erreur pendant le processus, reprends le traitement avec :

    python 011_resume_embeddings.py

### 3️⃣ Indexation FAISS et ajout des métadonnées

    Indexe les embeddings et ajoute les métadonnées dans FAISS :

    python 020_index_faiss_metadatas.py

### 4️⃣ Liaison FAISS + LangChain et tests locaux

    Teste le fonctionnement du système RAG en local :

    python 030_rag_langchain_faiss.py

### 5️⃣ Démarrage de la démo FastAPI

    Lance le serveur FastAPI avec :

    uvicorn 040_rag_fast_api:app --reload


Ensuite, ouvrez le navigateur à l’adresse suivante :

👉 http://127.0.0.1:8000/docs

### 6️⃣ Test des endpoints FastAPI

Sur l’interface Swagger (/docs), tu disposes de trois endpoints principaux 👇

| Endpoint   | Méthode         | Description                                                                                                     |
| ---------- | --------------- | --------------------------------------------------------------------------------------------------------------- |
| `/search`  | `POST`          | Recherche sémantique d’un événement                                                                             |
| `/ask`     | `POST`          | Génère une réponse détaillée et cohérente à propos d’un événement                                               |
| `/rebuild` | `POST` ou `GET` | Relance l’ensemble du pipeline : collecte des données, mise à jour des métadonnées, embeddings et index FAISS 
(pour garantir des données à jour) |


**🧠 Stack technique utilisée**

    LangChain – pour la gestion du pipeline RAG

    FAISS – pour l’indexation vectorielle et la recherche sémantique

    Mistral Embeddings – pour la création des représentations vectorielles

    FastAPI – pour l’exposition de l’API REST

    Python 3.11+ – langage principal du projet

    Poetry – pour la gestion des dépendances et environnements virtuels

**✅ Résultats attendus**

    Une API REST locale exposant le système RAG

    Un endpoint /ask qui retourne une réponse générée et contextualisée

    Un endpoint /rebuild permettant de reconstruire la base vectorielle à la demande

    Une documentation Swagger générée automatiquement

    Un test fonctionnel via un fichier api_test.py

















    
## 🎯 Objectif du projet (création d’un assistant intelligent pour recommander des événements culturels avec un système RAG)

**Creer un nouvel environnement virtuel python**

Étapes :

Ouvre ton dossier de projet dans VS Code.

Ouvre le terminal intégré :

Raccourci : Ctrl + ù (ou Ctrl + J)

ou menu : Affichage > Terminal

Tape la commande suivante :

python -m venv .venv


👉 Cela crée un dossier .venv (ou le nom que tu veux) contenant ton environnement virtuel.

Active-le selon ton système d’exploitation :

**Windows :**

.venv\Scripts\activate


**🍎 macOS / Linux :**

source .venv/bin/activate


(Optionnel) Vérifie que l’environnement est bien activé :

where python      # Windows
which python      # macOS/Linux


Tu dois voir le chemin pointant vers ton dossier .venv.

Installe ensuite tes dépendances :

pip install numpy pandas fastapi

4. 💻 Vérification finale sur une “installation propre” 

Tu peux simuler une nouvelle machine en exécutant : 

- poetry env remove python 
- poetry install 
- poetry run python test_imports.py



### ETAPES

Avant toute chose déplacez vous dans votre environnement et à la racine du projet

bash 

    C:\Users\...\Concevez_et_deployez_un_systeme_RAG>

    poetry shell (pour activer votre environnement virtuel)

    cd scripts

1- Collectez les données avec data_collected : ***python 000_data_collected.py***
2- Géneration des embeddings sur les colonnes de description : ***python 010_generate_embeddings.py*** 
    Pour éffectuer un test sur quelques lignes avant de faire sur l'ensemble switcher MODE_TEST en True et mettez le nombre de ligne à tester sur TEST_SIZE
    En cas de bug ou d'arrêt du code en cours, reprendre là où on en était : ***python 011_resume_embeddings.py***
3- Indexez les embeddings des nouvelles colonnes embéddées avec FAISS ET ajoutez les métadonnées : ***python 020_index_faiss_metadatas.py***

4- Faites la liaison entre les index et langchain pour la recherche sementique et lancez quelques tests en local : ***python 030_rag_langchain_faiss.py***

5- Lancez une démo sur FastAPI : ***uvicorn 040_rag_fast_api:app --reload*** 
        ouvrir le lien et rajouter "/docs" puis entrer
6- Sur l'interface FastAPI, testez les endpoints :
            - Search : pour éffectuer une recherche sémentique d'un évemenent
            - Ask : pour avoir une réponse cohérente et détaillée à propos d'un événement
            - Rebuild : pour relancer le process depuis la recupération des données sur CalandarEvents et la mise à jour des métadonnées, embeddings et index (afin de pouvoir travailler sur des données à jour)









