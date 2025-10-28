
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