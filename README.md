# 🚀 RAG Pipeline using Python, Qdrant & Groq

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that retrieves relevant information from a custom knowledge base using semantic search and generates context-aware answers using a Large Language Model (LLM).

## 📌 Project Overview

This project demonstrates the implementation of a RAG system by combining a vector database, embedding model, and LLM.

Instead of relying only on pre-trained knowledge, the system retrieves relevant information from external documents and provides it as context to the LLM to generate knowledge-based responses.

## 🛠️ Tech Stack

- **Python** – Core programming language
- **Qdrant Cloud** – Vector database for storing and retrieving embeddings
- **Sentence Transformers** – Text embedding generation
- **all-MiniLM-L6-v2** – Embedding model
- **Groq API** – LLM integration
- **GPT-OSS-120B** – Language model
- **Cosine Similarity** – Semantic similarity search

## ⚙️ How It Works

1. **Knowledge Loading:** Reads information from `knowledge.txt`.
2. **Embedding Generation:** Converts documents into 384-dimensional vectors using Sentence Transformers.
3. **Vector Storage:** Stores embeddings and original text in Qdrant Cloud.
4. **Query Processing:** Converts the user's question into an embedding.
5. **Semantic Search:** Uses cosine similarity to retrieve the Top-K relevant documents.
6. **Context Formation:** Extracts and combines the retrieved text.
7. **LLM Integration:** Sends the question and retrieved context to Groq.
8. **Answer Generation:** Generates a context-aware response.

## 🔄 RAG Architecture

```text
          Knowledge.txt
                |
                v
      Sentence Transformers
                |
                v
        Text Embeddings
                |
                v
          Qdrant Cloud
         (Vector Storage)
                |
                |
         User Question
                |
                v
         Query Embedding
                |
                v
         Semantic Search
        (Cosine Similarity)
                |
                v
          Top-K Retrieval
                |
                v
        Context Formation
                |
                v
             Groq
         (GPT-OSS-120B)
                |
                v
          Final Answer
```

## 📂 Project Structure

```text
RAG-Pipeline/
│
├── FULL_RAG.py
├── knowledge.txt
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 📦 Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd RAG-Pipeline
```

Install dependencies:

```bash
uv add qdrant-client sentence-transformers groq python-dotenv numpy
```

## 🔑 Environment Variables

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

**Important:** Never upload your `.env` file or expose API keys on GitHub.

## ▶️ Run the Project

```bash
python FULL_RAG.py
```

Make sure `knowledge.txt` contains your source information before running the application.

## 💡 Key Concepts Learned

- Retrieval-Augmented Generation (RAG)
- Text Embeddings
- Vector Databases
- Semantic Search
- Cosine Similarity
- Top-K Retrieval
- Context Formation
- LLM Integration
- Prompt Engineering

## 🚀 Future Improvements

- Document Chunking
- Hybrid Search
- Reranking
- Query Rewriting
- Metadata Filtering
- Advanced RAG Techniques

## 👨‍💻 Author

**Paras Mate**

Electronics & Computer Science Student | Exploring Generative AI, LLMs and Agentic AI.

---

⭐ If you find this project interesting, consider giving it a star!
