import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model (downloads ~90 MB the first time)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to Chroma running in Docker
client = chromadb.HttpClient(host="localhost", port=8001)
print("Chroma heartbeat:", client.heartbeat())

# 3. Create a collection that compares vectors by cosine similarity
col = client.get_or_create_collection("setup_test", metadata={"hnsw:space": "cosine"})

docs = [
    "Employees get 24 days of annual leave per year.",
    "The office is closed on public holidays.",
    "Laptops are replaced every three years.",
]
col.upsert(ids=["1", "2", "3"], documents=docs, embeddings=model.encode(docs).tolist())

# 4. Semantic search: "vacation" never appears in the docs
question = "How many vacation days do I get?"
res = col.query(query_embeddings=model.encode([question]).tolist(), n_results=1)
context = res["documents"][0][0]
print("Best match:", context, "| distance:", round(res["distances"][0][0], 3))

# 5. Ask the local LLM, grounded in the retrieved text
reply = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "Answer only using the given context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
    ],
    options={"temperature": 0.1},
)
print("LLM answer:", reply["message"]["content"])

client.delete_collection("setup_test")  # clean up