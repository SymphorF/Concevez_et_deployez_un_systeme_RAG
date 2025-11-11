# test_imports.py

try:
    import faiss
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from mistral import MistralClient
    print("✅ Tous les imports sont OK !")
except Exception as e:
    print("❌ Problème d'import :", e)
