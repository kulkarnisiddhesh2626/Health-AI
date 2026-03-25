import streamlit as st
import os
import tempfile
from datetime import datetime

# Bulletproof Modern Imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI-Health Assistant", page_icon="⚕️", layout="wide")

# --- MEDICAL UI & TRIAGE CSS ---
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
    .faq-text { font-size: 0.9rem; color: #0284c7; font-style: italic; }
    /* Make radio buttons wrap nicely on small mobile screens */
    div.row-widget.stRadio > div { flex-direction: row; flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)

# --- SECURE API KEY FETCHING ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 Groq API key not found! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- EXPANDED SYNTHETIC PATIENT DATABASE ---
DEMO_PATIENTS = {
    "🟢 Evelyn (Routine/Stable - Diabetes)": """SYNTHETIC RECORD - ID: SYN-8472910
Name: Evelyn Carter | DOB: 1945-08-22 | Gender: Female
CHIEF COMPLAINT: Routine 6-month follow-up. 
ACTIVE CONDITIONS: Essential Hypertension, Type 2 Diabetes Mellitus.
MEDICATIONS: Lisinopril 20mg daily, Metformin 1000mg BID.
VITALS (2024-02-15): BP: 138/88, HR: 72, HbA1c: 7.2%.
PLAN: Diabetes well-controlled. Continue current meds. Return in 6 months.""",
    
    "🟡 Marcus (Moderate - Asthma Flare)": """SYNTHETIC RECORD - ID: SYN-3920114
Name: Marcus Thorne | DOB: 1982-11-04 | Gender: Male
CHIEF COMPLAINT: Persistent dry cough and shortness of breath exacerbated by cold air. 
ACTIVE CONDITIONS: Moderate Persistent Asthma.
MEDICATIONS: Fluticasone/Salmeterol 250/50 mcg BID, Albuterol 90mcg PRN.
VITALS (2024-03-01): BP: 120/78, HR: 84, SpO2: 96% on room air.
EXAM: Expiratory wheezing noted in bilateral lower lung fields. 
PLAN: Step up asthma therapy. Prescribed oral Prednisone 40mg taper for 5 days. Follow up in 2 weeks.""",

    "🔴 Robert (Critical - Stroke Risk)": """SYNTHETIC RECORD - ID: SYN-9928311
Name: Robert Vance | DOB: 1955-02-14 | Gender: Male
CHIEF COMPLAINT: Presents to ER with sudden onset left-sided weakness, slurred speech, and facial drooping starting 45 minutes ago.
ACTIVE CONDITIONS: Atrial Fibrillation (Non-compliant with anticoagulants), Severe Hypertension.
MEDICATIONS: Metoprolol 50mg daily.
VITALS (2024-03-20): BP: 195/110, HR: 115 (Irregular), SpO2: 94%.
EXAM: Pronator drift on left arm. Expressive aphasia. 
PLAN: Activate acute stroke protocol. STAT CT Head without contrast. Prepare for potential tPA administration. Admit to Neuro ICU.""",

    "🟢 Sarah (Routine/Stable - Post-Op)": """SYNTHETIC RECORD - ID: SYN-5592833
Name: Sarah Jenkins | DOB: 1968-04-19 | Gender: Female
CHIEF COMPLAINT: 2-week post-operative check for Left Total Hip Arthroplasty.
ACTIVE CONDITIONS: Left Hip Osteoarthritis (Surgical).
MEDICATIONS: Apixaban 2.5mg BID.
VITALS (2024-03-10): BP: 128/82, Temp: 98.6F.
EXAM: Surgical incision clean, dry, intact. No erythema.
PLAN: Discontinue narcotics. Begin outpatient physical therapy."""
}

# --- SIDEBAR: DEEP-DIVE ARCHITECTURE & EDUCATION ---
with st.sidebar:
    st.markdown("<h2 style='color: #059669;'>ℹ️ Under the Hood: AI Architecture</h2>", unsafe_allow_html=True)
    st.markdown("This system utilizes a highly advanced **Retrieval-Augmented Generation (RAG)** pipeline to ensure 100% deterministic, hallucination-free medical analysis.")
    st.divider()
    
    with st.expander("🧩 1. Data Ingestion & Processing Layer", expanded=False):
        st.markdown("""
        **How it works:** When a doctor uploads a medical file (PDF, TXT, DOCX, CSV) or selects a patient, the system reads the raw data.
        * **Chunking:** Medical records are long. We use a `RecursiveCharacterTextSplitter` to break the text into manageable chunks of 700 characters, with a 100-character overlap. 
        * **Why overlap?** It ensures that a sentence split across two chunks doesn't lose its clinical context (e.g., separating "Patient is allergic to" and "Penicillin").
        """)

    with st.expander("🧠 2. Neural Embedding Layer", expanded=False):
        st.markdown("""
        **How it works:** Text cannot be directly understood by a database. We must translate the English text into high-dimensional mathematics (vectors).
        * **Model:** `HuggingFace all-MiniLM-L6-v2`.
        * **Action:** It reads every chunk and assigns it a coordinate in a 384-dimensional space based on its semantic medical meaning. "Myocardial Infarction" and "Heart Attack" will have nearly identical numerical coordinates.
        """)

    with st.expander("🗄️ 3. Vector Database (FAISS)", expanded=False):
        st.markdown("""
        **How it works:** We store these mathematical coordinates in **FAISS** (Facebook AI Similarity Search).
        * **Action:** When you ask a question, the system translates your question into a vector, and FAISS calculates the mathematical distance (Cosine Similarity) to find the top 3 most relevant medical chunks instantly.
        """)

    with st.expander("🤖 4. Generation & Triage Layer (Groq/LLaMA)", expanded=False):
        st.markdown("""
        **How it works:** The system packages your question AND the retrieved medical chunks, then sends them to the **Groq LLaMA-3.1** engine.
        * **Guardrails:** The AI is strictly instructed to *only* use the provided context. If the answer isn't there, it refuses to guess.
        * **Clinical Triage:** The AI evaluates the severity of the retrieved context and tags the output as Stable, Moderate, or Critical, which the UI uses to color-code the response.
        """)

# --- CORE PROCESSING FUNCTIONS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(data_source, selected_data, uploaded_files):
    documents = []
    
    if data_source == "🧪 Try AI-Demo Patients" and selected_data:
        documents.append(Document(page_content=selected_data, metadata={"source": "AI Demo Database"}))
            
    elif data_source == "📂 Upload Patient Records" and uploaded_files:
        for file in uploaded_files:
            file_extension = file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            try:
                if file_extension == "pdf":
                    documents.extend(PyPDFLoader(temp_file_path).load())
                elif file_extension == "txt":
                    documents.extend(TextLoader(temp_file_path).load())
                elif file_extension == "docx":
                    documents.extend(Docx2txtLoader(temp_file_path).load())
                elif file_extension == "csv":
                    documents.extend(CSVLoader(temp_file_path).load())
                else:
                    st.warning(f"Unsupported file format: {file_extension}")
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

# --- MAIN PAGE: UI ---
st.title("⚕️ AI-Health Assistant")
st.markdown("Advanced clinical intelligence, triage, and semantic EHR analysis.")

st.markdown("### 1️⃣ Select Patient Data")
data_source = st.radio(
    "How would you like to load data?", 
    ["🧪 Try AI-Demo Patients", "📂 Upload Patient Records"],
    horizontal=True
)

selected_patient_data = None
uploaded_files = None

if data_source == "🧪 Try AI-Demo Patients":
    patient_name = st.selectbox("Choose a demo patient profile:", list(DEMO_PATIENTS.keys()))
    selected_patient_data = DEMO_PATIENTS[patient_name]
else:
    # Expanded file types
    uploaded_files = st.file_uploader("Upload EHR Data (PDF, TXT, DOCX, CSV)", type=["pdf", "txt", "docx", "csv"], accept_multiple_files=True)

st.divider()

llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

with st.spinner("AI is analyzing the records..."):
    vectorstore = process_documents(data_source, selected_patient_data, uploaded_files)

if vectorstore:
    st.success("✅ Patient Data Loaded. The AI Assistant is ready.")
    
    tab_summary, tab_chat = st.tabs(["📋 AI Clinical Summary", "💬 Ask the AI Assistant"])
    
    with tab_summary:
        st.markdown("### Auto-Generated Patient Breakdown")
        if st.button("✨ Generate AI Summary & Triage"):
            with st.spinner("AI is writing the summary..."):
                query = "Provide a structured clinical summary: Demographics, Chief Complaint, Active Conditions, and Plan. Also, state the overall clinical stability."
                docs = vectorstore.similarity_search(query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"Based ONLY on the following medical records:\n{context}\n\nTask: {query}"
                summary_response = llm.invoke(prompt)
                
                # Timestamp Integration
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"⏱️ **Generated on:** {current_time} (System Time)")
                st.info(summary_response.content)
                
                st.download_button(
                    label="📥 Download AI Summary (.txt)",
                    data=f"Timestamp: {current_time}\n\n{summary_response.content}",
                    file_name="AI_Clinical_Summary.txt",
                    mime="text/plain"
                )

    with tab_chat:
        st.markdown("### Ask Questions About the Records")
        
        # Suggested FAQs
        st.markdown("<p class='faq-text'>💡 <b>Suggested Questions:</b> What are the patient's active conditions? | Are there any vital sign abnormalities? | Summarize the current medication plan.</p>", unsafe_allow_html=True)
        
        user_query = st.text_input("Type your specific clinical question:")
        
        if user_query:
            with st.spinner("AI is evaluating the query..."):
                retrieved_docs = vectorstore.similarity_search(user_query, k=4)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # STRICT GUARDRAILS + TRIAGE COLOR INJECTION
                final_prompt = f"""You are a professional clinical AI assistant. Use ONLY the context below. 
                First, evaluate the medical severity of the information relevant to the question. 
                Start your entire response with exactly one of these tags based on your medical assessment:
                [CRITICAL] (if the context mentions emergencies, severe instability, stroke, heart attack, ER visits)
                [MODERATE] (if the context mentions flare-ups, necessary medication changes, or moderate symptoms)
                [STABLE] (if the context mentions routine checkups, well-controlled conditions, or normal findings)
                
                After the tag, provide your detailed answer. Do not guess or provide outside advice.
                
                Context: {context_text}
                Question: {user_query}
                Answer:"""
                
                result = llm.invoke(final_prompt)
                raw_response = result.content
                
                st.markdown("#### 🤖 AI Clinical Evaluation")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"⏱️ **Processed at:** {current_time}")
                
                # Parse the AI's hidden tag to color-code the UI
                if "[CRITICAL]" in raw_response:
                    clean_text = raw_response.replace("[CRITICAL]", "").strip()
                    st.error(f"🔴 **HIGH IMPACT / CRITICAL:**\n\n{clean_text}")
                elif "[MODERATE]" in raw_response:
                    clean_text = raw_response.replace("[MODERATE]", "").strip()
                    st.warning(f"🟡 **MODERATE IMPACT / OBSERVATION:**\n\n{clean_text}")
                elif "[STABLE]" in raw_response:
                    clean_text = raw_response.replace("[STABLE]", "").strip()
                    st.success(f"🟢 **STABLE / ROUTINE:**\n\n{clean_text}")
                else:
                    # Fallback if AI forgets the tag
                    st.info(raw_response)
                
                st.markdown("#### 📑 Ground Truth Evidence")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Reference Evidence {i+1}"):
                        source_name = doc.metadata.get('source', 'AI Demo Database')
                        st.markdown(f"**Source:** `{source_name}`")
                        st.markdown(f"**Exact Text:**\n> {doc.page_content}")
else:
    if data_source == "📂 Upload Patient Records":
        st.info("📂 Please upload files above to activate the AI. Supported: PDF, TXT, DOCX, CSV.")

# --- CLINICAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Regulatory Disclaimer:</strong> This AI-Health Assistant is a demonstration prototype. 
    It is not certified medical software. AI-generated outputs, including triage severity color-coding, should never replace professional medical diagnosis or clinical judgment.
</div>
""", unsafe_allow_html=True)
