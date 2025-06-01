#DISPAYS THE TIME TAKEN TO GENERATE 5 EMBEDDINGS...
import requests
import numpy as np
import time

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "mxbai-embed-large"
TEXT_FILE = "vault.txt"

# Step 1: Load text
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Step 2: Simple chunking
def chunk_text(text, max_len=500):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        sentence += "."
        if len(chunk) + len(sentence) < max_len:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

chunks = chunk_text(text)

# Step 3: Get embeddings
def get_embedding(text):
    res = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": text
    })
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype=np.float32)

# Step 4: Embed all chunks
embeddings = [get_embedding(chunk) for chunk in chunks]

print(f"✅ Generated {len(embeddings)} embeddings.")

# Print first 5 embeddings
for i, emb in enumerate(embeddings[:5]):
    print(f"\nEmbedding {i+1}:\n{emb}\n")

print(f"✅ Generated {len(embeddings)} embeddings.")

# Step 6: time
start_time = time.time()

embeddings = [get_embedding(chunk) for chunk in chunks]

end_time = time.time()
elapsed = end_time - start_time

print(f"✅ Generated {len(embeddings)} embeddings in {elapsed:.2f} seconds.")
