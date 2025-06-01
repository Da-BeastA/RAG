# 🔍 LocalRAG: Retrieval-Augmented Generation (Fully Local)

LocalRAG is a fully local implementation of Retrieval-Augmented Generation (RAG) using:

- 🧠 [Ollama](https://ollama.com/) for local embedding and LLM inference
- 📚 [Weaviate](https://weaviate.io/) as the vector database
- ⚡ Fast, secure, and completely private (no internet calls)

## 🚀 Features
- Text chunking + semantic embedding
- Vector-based similarity search
- Natural language query answering
- All components run on your local machine

## 🛠 Tech Stack
- Ollama (`llama3`, `mxbai-embed-large`)
- Weaviate (local instance)
- Python 3.10+
- UUID, requests, and Weaviate client

## 🧪 Example Usage
```bash
python localrag.py

## Then enter your question interactively:
#- Ask something (or 'exit' to quit): What is this document about?

#To install the requirements:
#- pip install -r requirements.txt
