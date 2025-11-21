# streamlit_app.py
import streamlit as st
import requests
import time

# Configuration de la page
st.set_page_config(
    page_title="Chat RAG - Événements Culturels",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .assistant-message {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ecdc4;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"  # Ajustez si nécessaire

class RAGClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def ask_question(self, question, k=5):
        """Pose une question au système RAG"""
        try:
            response = requests.post(
                f"{self.base_url}/ask",
                json={"question": question, "k": k},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Erreur API: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez que le serveur FastAPI est démarré sur le port 8000.")
            return None
        except Exception as e:
            st.error(f"Erreur de connexion: {e}")
            return None
    
    def health_check(self):
        """Vérifie l'état de l'API"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Initialisation du client
client = RAGClient(API_BASE_URL)

def main():
    # En-tête principale
    st.markdown('<h1 class="main-header">💬 Chat RAG - Événements Culturels</h1>', unsafe_allow_html=True)
    
    # Vérification de la santé de l'API
    st.subheader("État du système")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if client.health_check():
            st.success("✅ API connectée")
            status = "connectée"
        else:
            st.error("❌ API non disponible")
            status = "non disponible"
    
    with col2:
        st.info("""
        **Posez des questions sur les événements culturels :**
        - Concerts, expositions, spectacles
        - Ateliers, conférences, festivals
        - Recherche par ville ou par date
        """)
    
    if status == "non disponible":
        st.warning("""
        **Pour démarrer l'API :**
        1. Ouvrez un terminal
        2. Naviguez vers le dossier de votre projet
        3. Exécutez : `python rag_fast_api.py`
        4. Attendez que le message "Application startup complete" s'affiche
        5. Rechargez cette page
        """)
        return
    
    # Paramètres dans la sidebar
    with st.sidebar:
        st.header("⚙️ Paramètres")
        k_results = st.slider("Nombre de documents utilisés", min_value=1, max_value=10, value=5)
        
        st.header("💡 Exemples de questions")
        example_questions = [
            "Quels concerts de jazz à Paris ce week-end ?",
            "Y a-t-il des expositions photo à Lyon ?",
            "Quels ateliers pour enfants à Bordeaux ?",
            "Donne-moi les événements de danse à Marseille",
            "Quels sont les festivals ce mois-ci ?"
        ]
        
        for question in example_questions:
            if st.button(f"🗨️ {question}", key=question, use_container_width=True):
                if "messages" in st.session_state:
                    st.session_state.messages.append({"role": "user", "content": question})
                    # Déclencher le traitement
                    st.session_state.process_question = question
        
        st.markdown("---")
        if st.button("🗑️ Effacer l'historique", type="secondary"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    # Initialisation de l'historique de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Bonjour ! Je suis votre assistant pour les événements culturels. Posez-moi vos questions sur les concerts, expositions, spectacles, ateliers, et bien plus !"
            }
        ]
    
    # Affichage de l'historique des messages
    st.markdown("### 💭 Conversation")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>Vous:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Input utilisateur
    st.markdown("### 💬 Votre question")
    if prompt := st.chat_input("Ex: Quels concerts à Paris ce week-end ?"):
        # Ajout du message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Affichage immédiat du message utilisateur
        st.markdown(f'<div class="user-message"><strong>Vous:</strong> {prompt}</div>', unsafe_allow_html=True)
        
        # Génération de la réponse
        with st.spinner("🔍 Recherche d'événements et génération de réponse..."):
            response = client.ask_question(prompt, k=k_results)
        
        if response and "generated_answer" in response:
            answer = response["generated_answer"]
            
            # Affichage de la réponse
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {answer}</div>', unsafe_allow_html=True)
            
            # Informations supplémentaires dans un expander
            with st.expander("📊 Détails techniques de la recherche"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Ville détectée :** {response.get('city_detected', 'Aucune')}")
                    st.write(f"**Documents utilisés :** {response.get('results_used', 0)}")
                with col2:
                    st.write(f"**Modèle :** Mistral")
                    st.write(f"**Statut :** ✅ Réponse générée")
            
            # Ajout de la réponse à l'historique
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        else:
            error_msg = "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer ou vérifier la connexion à l'API."
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {error_msg}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

def display_simple_chat():
    """Version alternative ultra-simplifiée"""
    st.title("💬 Chat RAG Événements")
    
    # Vérification API
    if not client.health_check():
        st.error("❌ API non disponible - Démarrez d'abord rag_fast_api.py")
        return
    
    # Historique
    if "chat" not in st.session_state:
        st.session_state.chat = []
    
    # Affichage historique
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.write(f"**Vous:** {msg['content']}")
        else:
            st.write(f"**Assistant:** {msg['content']}")
        st.divider()
    
    # Input
    question = st.text_input("Posez votre question sur les événements culturels:")
    if st.button("Envoyer") and question:
        st.session_state.chat.append({"role": "user", "content": question})
        
        with st.spinner("Recherche en cours..."):
            response = client.ask_question(question)
        
        if response and "generated_answer" in response:
            answer = response["generated_answer"]
            st.session_state.chat.append({"role": "assistant", "content": answer})
            st.rerun()
        else:
            st.error("Erreur de génération")

if __name__ == "__main__":
    main()
    # Pour la version ultra-simple, décommentez la ligne suivante :
    # display_simple_chat()