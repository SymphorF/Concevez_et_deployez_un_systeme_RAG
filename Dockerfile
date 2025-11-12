# ============================================================
# Étape 1 — Construction de l’environnement
# ============================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============================================================
# 🚀 Étape 2 — Image finale légère
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Copie uniquement ce qui est nécessaire depuis la première image
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie le code de l’application
COPY . .

# CRÉATION DES DOSSIERS ET COPIE DES INDEX FAISS
RUN mkdir -p /app/data/processed/faiss_indexes && \
    cp -r /app/scripts/data/processed/faiss_indexes/* /app/data/processed/faiss_indexes/ && \
    echo "✅ Index FAISS copiés vers /app/data/processed/faiss_indexes/"

# Vérification que les fichiers sont bien copiés
RUN ls -la /app/data/processed/faiss_indexes/ && \
    echo "📁 Fichiers index présents:" && \
    find /app/data/processed/faiss_indexes/ -name "*.index" -o -name "*.pkl" | head -10

# Expose le port FastAPI/Uvicorn
EXPOSE 8001 

# Variables d'environnement pour l’application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001 \
    PYTHONPATH=/app/scripts

CMD ["uvicorn", "scripts.rag_fast_api:app", "--host", "0.0.0.0", "--port", "8001"]