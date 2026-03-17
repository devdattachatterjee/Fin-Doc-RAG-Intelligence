# 🏦 Fin-Doc RAG Intelligence
### *Enterprise-Grade Financial Document Analysis via Custom RAG Architecture*

[![Live Application](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://fin-doc-rag-intelligence-rno99koelwuf8pmtgf6oos.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT_4o_mini-412991?style=for-the-badge&logo=openai)](https://openai.com/)

## 🚀 Project Overview
Fin-Doc RAG Intelligence is a specialized Generative AI application designed for the financial sector. It allows users to securely upload complex, multi-page regulatory filings, annual reports, or circulars and query them with absolute factual grounding. 

Unlike standard AI wrappers, this project implements a **Framework-Free Retrieval-Augmented Generation (RAG) pipeline**. By bypassing heavy abstraction layers like LangChain or LlamaIndex, the system achieves lower latency, higher transparency, and complete control over the semantic chunking and vector retrieval mechanics.

## 🧠 Architectural Highlights
* **Framework-Free RAG:** Custom-built text chunking and overlap logic natively in Python to handle complex financial data without framework-induced memory bloat.
* **In-Memory Vector Search:** Utilizes **FAISS (Facebook AI Similarity Search)** via C++ bindings for lightning-fast L2 distance calculations across document embeddings.
* **Semantic Embeddings:** Powered by OpenAI's `text-embedding-3-small` for highly nuanced semantic capture of financial terminology.
* **Enterprise UI/UX:** Custom CSS implementation overriding default Streamlit aesthetics to provide a dark-mode, production-ready SaaS interface.

## 🛠️ Technical Stack
* **Core Logic:** Pure Python, NumPy
* **Vector Database:** FAISS (`faiss-cpu`)
* **LLM Engine:** OpenAI API (`gpt-4o-mini`)
* **Document Processing:** `pypdf`
* **Frontend:** Streamlit

## ⚙️ System Workflow
1. **Ingestion:** User securely uploads a PDF document. 
2. **Parsing & Chunking:** The system extracts raw text and applies a custom overlapping window algorithm (Chunk size: 1000, Overlap: 150) to preserve contextual boundaries.
3. **Embedding:** Text chunks are vectorized using OpenAI's embedding models.
4. **Indexing:** Vectors are stored in a highly optimized FAISS IndexFlatL2 database.
5. **Retrieval & Generation:** User queries are embedded, matched against the FAISS index using K-Nearest Neighbors (k=3), and injected into a strict prompt for GPT-4o-mini to synthesize a hallucination-free answer.

