import streamlit as st
import os
import tempfile
import base64

# Bulletproof Modern Imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Asclepius.AI | Clinical Intelligence", page_icon="⚕️", layout="wide", initial_sidebar_state="expanded")

# --- MEDICAL GREEN CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button {
        background-color: #059669; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; font-weight: 600; transition: all 0.3s ease; width: 100%;
    }
    .stButton>button:hover {
        background-color: #047857; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transform: translateY(-1px); color: white;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 10px; border-left: 4px solid #059669;
    }
    .disclaimer {
        font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- SECURE API KEY FETCHING ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 Groq API key not found! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- SYNTHETIC PATIENT DATABASE (No external files needed) ---
DEMO_PATIENTS = {
    "Evelyn Carter (Cardiology & Endo)": """SYNTHETIC PATIENT RECORD - ID: SYN-8472910
Name: Evelyn Carter | DOB: 1945-08-22 | Gender: Female
CHIEF COMPLAINT: Routine 6-month follow-up. Reports occasional mild dizziness when standing.
ACTIVE CONDITIONS: Essential Hypertension (2012), Type 2 Diabetes Mellitus (2015), Osteoarthritis of right knee (2019).
MEDICATIONS: Lisinopril 20mg daily, Metformin 1000mg BID, Ibuprofen 400mg PRN.
VITALS (2024-02-15): BP: 138/88 mmHg, HR: 72 bpm, Weight: 165 lbs, HbA1c: 7.2%, Fasting Glucose: 130 mg/dL.
PLAN: Continue Lisinopril. Diabetes well-controlled. Counselled on orthostatic hypotension. Hydration advised.""",
    
    "Marcus Thorne (Pulmonology)": """SYNTHETIC PATIENT RECORD - ID: SYN-3920114
Name: Marcus Thorne | DOB: 1982-11-04 | Gender: Male
CHIEF COMPLAINT: Presents with persistent dry cough and shortness of breath exacerbated by exercise and cold air. 
ACTIVE CONDITIONS: Moderate Persistent Asthma (Diagnosed 1995), Seasonal Allergic Rhinitis.
MEDICATIONS: Fluticasone/Salmeterol 250/50 mcg inhaler BID, Albuterol 90mcg PRN, Cetirizine 10mg daily.
VITALS (2024-03-01): BP: 120/78 mmHg, HR: 84 bpm, SpO2: 96% on room air.
EXAM: Expiratory wheezing noted in bilateral lower lung fields. 
PLAN: Step up asthma therapy. Prescribed oral Prednisone 40mg taper for 5 days. Follow up in 2 weeks.""",

    "Sarah Jenkins (Post-Op Ortho)": """SYNTHETIC PATIENT RECORD - ID: SYN-5592833
Name: Sarah Jenkins | DOB: 1968-04-19 | Gender: Female
CHIEF COMPLAINT: 2-week post-operative check for Left Total Hip Arthroplasty (THA).
ACTIVE CONDITIONS: Left Hip Osteoarthritis (Surgical), Hyperlipidemia.
MEDICATIONS: Atorvastatin 20mg daily, Apixaban 2.5mg BID (DVT prophylaxis), Acetaminophen 1000mg TID.
VITALS (2024-03-10): BP: 128/82 mmHg, HR: 76 bpm, Temp: 98.6F.
EXAM: Surgical incision site clean, dry, intact. No erythema or purulent drainage. Mild edema in left lower extremity. Range of motion improving.
PLAN: Discontinue narcotic pain relievers. Continue Apixaban for 2 more weeks. Begin outpatient physical therapy twice weekly."""
}

# --- SIDEBAR: ARCHITECTURE & CONTROLS ---
with st.sidebar:
    st.markdown("<h1 style='color: #059669;'>⚕️ Asclepius.AI</h1>", unsafe_allow_html=True)
    st.markdown("*Enterprise Clinical Intelligence*")
    
    st.divider()
    
    st.subheader("📊 Patient Data Pipeline")
    data_source = st.radio(
        "Select Pipeline Input:", 
        ["🧬 Load Synthetic Cohort (Demo)", "📂 Upload Medical Records"]
    )
    
    selected_patient_data = None
    uploaded_files = None
    
    if data_source == "🧬 Load Synthetic Cohort (Demo)":
        patient_name = st.selectbox("Select Patient Profile:", list(DEMO_PATIENTS.keys()))
        selected_patient_data = DEMO_PATIENTS[patient_name]
    else:
        uploaded_files = st.file_uploader("Upload EHR Data (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

    st.divider()
    
    # --- WOW FACTOR: ARCHITECTURE & SCOPE ---
    with st.expander("⚙️ System Architecture & Flow"):
        st.markdown("**Semantic Retrieval-Augmented Generation (RAG)**")
        # Native Streamlit Graphviz Engine
        st.graphviz_chart('''
            digraph {
                bgcolor="transparent"
                node [shape=box, style=filled, fillcolor="#ecfdf5", color="#059669", fontname="Helvetica", fontsize=10]
                edge [color="#047857", arrowsize=0.7]
                A [label="Data Ingestion\n(PDF/TXT/Mock)"]
                B [label="Recursive Text\nChunking (700t/100o)"]
                C [label="HuggingFace\nMiniLM Embeddings"]
                D [label="FAISS Vector\nDatabase"]
                E [label="Groq LLaMA-3.1\nInference Engine"]
                F [label="Clinical Output\n& Citations"]
                A -> B -> C -> D
                D -> E [label=" Semantic Search", fontsize=9, fontcolor="#059669"]
                E -> F
            }
        ''')
    
    with st.expander("🛠️ Tech Stack & Enhancements"):
        st.markdown("""
        * **LLM Engine:** Groq (LLaMA-3.1-8B-Instant) for ultra-low latency inference.
        * **Vector Storage:** FAISS (Facebook AI Similarity Search) for dense vector indexing.
        * **Embeddings:** `all-MiniLM-L6-v2` for semantic meaning extraction.
        * **Guardrails:** Deterministic prompt constraints to prevent medical hallucinations.
        """)
        
    with st.expander("📋 Scope & Limitations"):
        st.markdown("""
        * **Scope:** Instantly parses complex EHR data, extracts vitals, summarizes chronologies, and cites sources.
        * **Limitation:** Does not predict future conditions (Diagnostics).
        * **Limitation:** Constrained entirely to the context window (Zero external hallucinations allowed).
        """)

# --- CORE PROCESSING FUNCTIONS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(data_source, selected_data, uploaded_files):
    documents = []
    
    if data_source == "🧬 Load Synthetic Cohort (Demo)" and selected_data:
        documents.append(Document(page_content=selected_data, metadata={"source": "Synthetic EHR Database"}))
            
    elif data_source == "📂 Upload Medical Records" and uploaded_files:
        for file in uploaded_files:
            file_extension = file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            try:
                if file_extension.lower() == "pdf":
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())
                elif file_extension.lower() == "txt":
                    loader = TextLoader(temp_file_path)
                    documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
            finally:
                os.unlink(temp_file_path)
                
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# --- MAIN APPLICATION UI ---
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

st.title("⚕️ Asclepius.AI | Advanced Patient Intelligence")
st.markdown("Semantic analysis and conversational retrieval for electronic health records.")

with st.spinner("Initializing Vector Space and Embedding Medical Data..."):
    vectorstore = process_documents(data_source, selected_patient_data, uploaded_files)

if vectorstore:
    st.success("✅ Neural Indexing Complete. Ready for Clinical Query.")
    
    tab_summary, tab_chat = st.tabs(["📋 Executive Clinical Summary", "💬 AI Diagnostic Assistant"])
    
    with tab_summary:
        st.markdown("### Auto-Generated Patient Breakdown")
        if st.button("✨ Generate Exhaustive Medical Summary"):
            with st.spinner("Compiling comprehensive summary..."):
                query = "Provide a highly detailed, structured clinical summary. Include sections for: Patient Demographics, Chief Complaint, Active Conditions, Medications, Vitals, and Plan. Use clear bullet points and bold headers."
                docs = vectorstore.similarity_search(query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"Based ONLY on the following medical records:\n{context}\n\nTask: {query}"
                summary_response = llm.invoke(prompt)
                
                st.info(summary_response.content)
                
                # WOW FACTOR: Export Button
                st.download_button(
                    label="📥 Export Summary to TXT",
                    data=summary_response.content,
                    file_name="clinical_summary.txt",
                    mime="text/plain"
                )

    with tab_chat:
        st.markdown("### Interrogate EHR Data")
        user_query = st.text_input("Ask a specific, highly detailed question about the patient's health, vitals, or history:")
        
        if user_query:
            with st.spinner("Executing Semantic Search across Vector Database..."):
                retrieved_docs = vectorstore.similarity_search(user_query, k=3)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # EXTREMELY STRICT AND DETAILED GUARDRAILS
                final_prompt = f"""You are Asclepius, an expert, highly professional clinical AI assistant. 
                Use ONLY the following pieces of context to answer the user's question. Provide a highly detailed, exhaustive response based strictly on the data.
                Do not provide general medical advice outside of the context. If the answer is not explicitly contained within the context, state clearly: "I cannot answer this based on the provided medical records."
                
                Context: {context_text}
                
                Question: {user_query}
                
                Professional Clinical Answer:"""
                
                result = llm.invoke(final_prompt)
                
                st.markdown("#### 🤖 Asclepius AI Response")
                st.success(result.content)
                
                st.markdown("#### 📑 Ground Truth Citations")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Reference Document Vector {i+1}"):
                        source_name = doc.metadata.get('source', 'Synthetic EHR Database')
                        st.markdown(f"**Source Data:** `{source_name}`")
                        st.markdown(f"**Extracted Text:**\n> {doc.page_content}")
else:
    if data_source == "📂 Upload Medical Records":
        st.info("📂 Please upload PDF or TXT files in the sidebar to begin neural indexing.")

# --- CLINICAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Regulatory Disclaimer:</strong> Asclepius.AI is a demonstration prototype designed for educational and architectural portfolio purposes. 
    It is not an FDA-certified medical device. The AI-generated outputs, semantic search results, and vector embeddings should never be used as a substitute for professional medical diagnosis or treatment.
</div>
""", unsafe_allow_html=True)
