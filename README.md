# 🧠 DocMind AI

## Secure RAG‑Powered Document Intelligence Platform

<p align="left">
  <b>Enterprise‑grade Retrieval‑Augmented Generation (RAG)</b><br/>
  Multi‑PDF • Secure • Explainable • Production‑oriented
</p>

---

## 🚀 What is DocMind AI?

**DocMind AI** is a **secure, production‑ready Retrieval‑Augmented Generation (RAG) system** that allows users to upload multiple PDF documents and ask natural‑language questions with **answers grounded strictly in document context**.

This project is intentionally designed to look, feel, and behave like a **real internal enterprise AI system**, not a demo chatbot.

---

## 🎯 Why Recruiters Love This Project

✅ Real RAG architecture (not prompt‑stuffing)
✅ Vector database with Pinecone (serverless)
✅ LLaMA‑3.1 inference via Groq (low latency)
✅ Secure authentication & role‑based access
✅ Explainable AI with source attribution
✅ Cost‑aware design (caching + hashing)

> 💡 This project demonstrates **LLM system engineering**, not just API usage.

---

## 🧠 Live Architecture Overview

```text
User
 ↓
Streamlit UI
 ↓
Authentication Layer (SHA‑256)
 ↓
PDF Upload
 ↓
Text Chunking (Recursive)
 ↓
Embeddings (MiniLM)
 ↓
Pinecone Vector DB (Serverless)
 ↓
Context Retrieval (Top‑K)
 ↓
LLaMA‑3.1 via Groq
 ↓
Grounded Answer + Sources
```

---

## ✨ Feature Showcase

### 📚 Multi‑PDF RAG

Upload **multiple PDFs** and query them simultaneously.

### 🔍 Semantic Search

Dense vector similarity using **Pinecone**.

### 📌 Explainable AI

Every answer includes:

* Source document
* Page number
* Content preview

### 🔐 Secure by Design

* Role‑based authentication
* Password hashing (SHA‑256)
* Environment‑based secrets

### ♻️ Smart Caching

* File hashing prevents re‑embedding
* Faster queries
* Reduced cost

---

## ⚙️ Tech Stack

| Layer      | Technology              |
| ---------- | ----------------------- |
| UI         | Streamlit               |
| LLM        | LLaMA‑3.1‑8B (Groq)     |
| Framework  | LangChain               |
| Embeddings | MiniLM‑L6‑v2            |
| Vector DB  | Pinecone (Serverless)   |
| Docs       | PyPDFLoader             |
| Auth       | SHA‑256 + Session State |

---

## 🔐 Access & Demo Policy (Important)

> ⚠️ **Pinecone vector storage is intentionally restricted**

This project uses a **persistent Pinecone index**, which:

* Incurs cost
* Stores embedded document data
* Is shared across sessions

### Therefore:

❌ Pinecone credentials are **not public**
❌ Open deployment is **intentionally disabled**
✅ **Live demo access is provided on request**

---

## 📩 Request a Demo / Interview Walkthrough

If you are a **recruiter, interviewer, or hiring manager**, I can:

* Grant **temporary Pinecone access**
* Walk through **architecture & design choices**
* Explain **scalability, cost, and security trade‑offs**
* Demonstrate **real‑time document intelligence**

### 👤 Author

**Yash Handa**
📧 Email: *hyash2455@gmail.com*
🔗 LinkedIn: *www.linkedin.com/in/yashhanda18*

> 🔒 Access is provided strictly for evaluation and interview purposes.

---

## 🚧 Future Enhancements

* Multi‑tenant namespaces
* Metadata‑aware retrieval
* Hybrid (BM25 + dense) search
* Streaming responses
* OCR for scanned PDFs
* RBAC per document
