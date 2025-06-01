from numpy import vectorize
import requests
import weaviate
from weaviate.classes.config import Configure, DataType
import uuid

# === Config ===
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434/api/embeddings"

WEAVIATE_CLASS = "DocumentChunk"

# Initialize Weaviate client
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",     # <-- dummy, required as string
    grpc_port=50051,           # <-- dummy, required as int
    grpc_secure=False,
    skip_init_checks=True      # <-- skips gRPC ping
)

def create_schema():
    existing_collections = client.collections.list_all()

    if WEAVIATE_CLASS in existing_collections:
        print(f"⚠️ Collection '{WEAVIATE_CLASS}' already exists. Skipping creation.")
        return

    print(f"✅ Creating collection '{WEAVIATE_CLASS}'")
    client.collections.create(
        name=WEAVIATE_CLASS,
        properties=[
            property(name="text", data_type=DataType.TEXT),
        ],
        vectorizer_config=vectorize.none(),
    )

def chunk_text(text, max_length=500):
    chunks, chunk = [], ""
    for sentence in text.split(". "):
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def get_embedding(text_chunk):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text_chunk}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def main():
    create_schema()

    with open("vault.txt", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Chunked text into {len(chunks)} chunks. Generating embeddings and uploading...")

    collection = client.collections.get(WEAVIATE_CLASS)

    for idx, chunk in enumerate(chunks, start=1):
        embedding = get_embedding(chunk)
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"docchunk-{idx}"))
        collection.data.insert(
            properties={"text": chunk},
            vector=embedding,
            uuid=uid
        )
        print(f"Uploaded chunk {idx}/{len(chunks)}")

    print(f"✅ Uploaded all {len(chunks)} chunks.")
    client.close()

if __name__ == "__main__":
    main()
    client.close()
