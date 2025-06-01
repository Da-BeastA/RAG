import uuid
import requests
import weaviate
from weaviate.classes.config import DataType

# === Config ===
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_COMPLETION_MODEL = "mistral-7b-chat"  # Change if needed
OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
OLLAMA_COMPLETION_URL = "http://localhost:11434/api/completions"

WEAVIATE_CLASS = "DocumentChunk"

# Initialize Weaviate client
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8081,
    http_secure=False,
    grpc_host="localhost",     # dummy
    grpc_port=50051,           # dummy
    grpc_secure=False,
    skip_init_checks=True
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
            dict(name="text", data_type=DataType.TEXT),
        ],
        vectorizer_config=None,
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
        OLLAMA_EMBEDDING_URL,
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text_chunk}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def create_and_upload():
    create_schema()

    with open("vault.txt", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Chunked text into {len(chunks)} chunks. Generating embeddings and uploading...")

    collection = client.collections.get(WEAVIATE_CLASS)

    for idx, chunk in enumerate(chunks, start=1):
        embedding = get_embedding(chunk)
        uid = str(uuid.uuid4())
        collection.data.insert(
            properties={"text": chunk},
            vector=embedding,
            uuid=uid
        )
        print(f"Uploaded chunk {idx}/{len(chunks)}")

    print(f"✅ Uploaded all {len(chunks)} chunks.")

def query_weaviate(query_text, top_k=3):
    # Get query embedding
    query_embedding = get_embedding(query_text)
    collection = client.collections.get(WEAVIATE_CLASS)

    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_properties=["text"]
    )

    # Extract the texts from results
    chunks = [obj.properties["text"] for obj in response.objects]
    return chunks


def get_ollama_response_cli(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_COMPLETION_MODEL, prompt],
        capture_output=True,
        text=True
    )
    return result.stdout

def answer_query(query):
    relevant_chunks = query_weaviate(query)
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = get_ollama_response(prompt)
    return answer

def main():
    create_and_upload()
    # Uncomment to create schema and upload chunks once
    # create_and_upload()

    while True:
        user_query = input("\nAsk something (or 'exit' to quit): ")
        if user_query.strip().lower() in {"exit", "quit"}:
            break
        
        answer = answer_query(user_query)
        print(f"\nOllama says:\n{answer}")

    client.close()

if __name__ == "__main__":
    main()
    client.close()
