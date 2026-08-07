FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# L'image locale fonctionne sur CPU. Installer PyTorch depuis son dépôt CPU
# évite d'embarquer plusieurs gigaoctets de bibliothèques NVIDIA inutilisées.
RUN pip install --no-cache-dir \
        torch==2.9.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
