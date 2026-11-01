
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

try:
    print(client.get_collections())
    print("Qdrant Connected Successfully!")

except Exception as e:
    print("Qdrant Error:", e)
    raise

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("Groq Connected Successfully!")

COLLECTION_NAME = "Knowledge"
EMBEDDING_SIZE = 384

if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing Collection {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    ),
)

print(f"Created Collection: {COLLECTION_NAME}")
print(f"Vector Size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

with open("knowledge", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

if not documents:
    raise ValueError("knowledge.txt is empty!")

print(f"Loaded {len(documents)} Documents")

print("Loading Embedding Model......")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Model Loaded")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")

points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i + 1,
        vector=embedding.tolist(),
        payload={
            "text": documents[i]
        }
    )
    points.append(point)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} points to Qdrant")


def search(query, top_k=3):
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results


query = "How many vacation days do I get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()


def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


question = "How many vacation days do I get?"

results = search(question, top_k=3)

context = "\n".join(
    result.payload["text"]
    for result in results
)

answer = ask_llm(question, context)

print("\nFinal Answer:")
print(answer)
