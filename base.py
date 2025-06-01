import requests

# Config
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434/api/embeddings"

# Load your text
with open("vault.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Simple chunking
def chunk_text(text, max_length=500):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

chunks = chunk_text(text)

# Get embeddings via Ollama
def get_embedding(text_chunk):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_EMBEDDING_MODEL,
        "prompt": text_chunk
    })
    response.raise_for_status()
    return response.json()["embedding"]

embeddings = [get_embedding(chunk) for chunk in chunks]

print(f"✅ Generated {len(embeddings)} embeddings.")