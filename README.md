# Assistant intelligent de recommandation d’événements culturels (Système RAG)

## Objectif du projet

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

    Avant toute manipulation, assurez-vous d’être dans ton environnement virtuel et à la racine du projet.

    C:\Users\...\Concevez_et_deployez_un_systeme_RAG>

    poetry shell       # Active ton environnement virtuel

    cd scripts


### 1️⃣ Collecte des données

    Collecte les données brutes depuis les sources externes :

    python data_collected.py

### 2️⃣ Génération des embeddings

    Génère les embeddings sur les colonnes de description :

    python generate_embeddings.py


***💡 Astuce :***

    Pour tester sur un échantillon limité, mettez MODE_TEST = True

    Défini TEST_SIZE pour le nombre de lignes à traiter

    En cas d’arrêt ou d’erreur pendant le processus, reprennez le traitement avec :

    python resume_embeddings.py

### 3️⃣ Indexation FAISS et ajout des métadonnées

    Indexez les embeddings et ajoutez les métadonnées dans FAISS :

    python index_faiss_metadatas.py

### 4️⃣ Liaison FAISS + LangChain et tests locaux

    Testez le fonctionnement du système RAG en local :

    python rag_langchain_faiss.py

### 5️⃣ Démarrage de la démo FastAPI

    Lancez le serveur FastAPI avec :

    uvicorn rag_fast_api:app --reload


Ensuite, ouvrez le navigateur à l’adresse suivante :

👉 http://127.0.0.1:8000/docs

### 6️⃣ Test des endpoints FastAPI

Sur l’interface Swagger (/docs), vous disposez de cinq endpoints principaux 👇

| Endpoint   | Méthode         | Description                                                                                                         |
| ---------- | --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `/`        | `GET`           | Message d'acceuil du RAG                                                                                            |
| `/healt`   | `GET`           | Vérifie l'état de l'API                                                                                             |
| `/search`  | `GET`           | Recherche sémantique d’un événement                                                                                 |
| `/ask`     | `POST`          | Génère une réponse détaillée et cohérente à propos d’un événement                                                   |
| `/rebuild` | `POST`          | Relance l’ensemble du pipeline : collecte des données, mise à jour des métadonnées, embeddings et index FAISS (pour garantir des données à jour) |


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

### 7. Docker

Voici le workflow résumé :

- Builder l’image Docker (crée l’image avec l'application et ses dépendances) :

docker build -t rag_api .

- Lancer un conteneur à partir de l’image (exécuter l'app en arrière-plan, mapper le port 8000 du conteneur vers le PC) :

docker run -d -p 8001:8001 --name rag_container rag_api:latest

- Accéder à l’API via le navigateur (FastAPI fournit automatiquement la documentation interactive Swagger) :

http://localhost:8001/docs (bien sûre en utilisant le post correct pour visualiser, dans ce exple c'est le port 8000)


**💡 Astuce :**

***Pour voir toutes les commandes docker***

docker

***Pour vérifier les images existantes***

docker images

***Pour supprimer l'image par son ID***

docker rmi 64c54753a78a (son ID)

***OU par son nom et tag***

docker rmi fastapirag-app:latest (son tag)

***Si l'image est utilisée par un conteneur (même arrêté), forcer la suppression***

docker rmi -f 64c54753a78a (son ID)

***Pour inspecter les logs du conteneur pour voir ce qui se passe :***

docker logs -f fastapi-app

***Pour visualiser la liste des contenair et les ports déjà utilisé par docker***

docker ps

***Pour arrêter un conteneur en particulier***

docker stop nom_du_conteneur (exp: docker stop eager_jemison)
docker rm nom_du_conteneur (exp: docker rm eager_jemison)

***Pour arrêter tous les conteneurs en même temps:***

docker stop $(docker ps -q)

***Pour supprimer tous les conteneurs (libérer les ports):***

docker rm $(docker ps -aq)

***Pour nettoyer tout le système Docker (arrêter tous les conteneurs, toutes les images non utilisées...):***

docker system prune -a

***Pour visualiser l'ensemble des images créées sur docker***

docker images

***Pour supprimer une image***

docker rmi id_image (exp docker rmi c111c74738e7)

Pensez à supprimer d'abord le conteneur utilisant cette image avant de la supprimer (voir méthode ci-dessus)