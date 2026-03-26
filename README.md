# ⚕️ AI-Health Assistant | Clinical Intelligence RAG System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-orange)

An enterprise-grade, multi-modal Retrieval-Augmented Generation (RAG) prototype designed for healthcare. This system ingests electronic health records (EHRs) across various formats, vectorizes the clinical data, and leverages Groq's LLaMA-3.1 engine to provide zero-hallucination medical timelines, clinical summaries, and automated triage evaluations.

## 🚀 Key Features
* **Multi-Modal Ingestion:** Supports PDF, TXT, DOCX, CSV, and Images (PNG, JPG) via Tesseract OCR.
* **Automated Clinical Triage:** Evaluates patient data and automatically tags output severity (🟢 Stable, 🟡 Moderate, 🔴 Critical).
* **Chronological Timelining:** Extracts dates from unstructured text to build accurate chronological patient histories.
* **Deterministic Guardrails:** Strict prompt engineering ensures the AI *only* uses uploaded context, preventing medical hallucinations.
* **Source Citations:** Transparently displays the exact document excerpts used to formulate responses.

## 🧠 System Architecture

```mermaid
graph TD;
    A[Raw Patient Data: PDFs, Images, DOCX] --> B(OCR & Text Chunking);
    B --> C[HuggingFace MiniLM Embeddings];
    C --> D[(FAISS Vector Database)];
    D <-->|Semantic Search| E{Groq LLaMA-3.1 Engine};
    E -->|Triage & Timeline| F[Clinical UI Output];
