import streamlit as st
import os
import tempfile
import requests

# Bulletproof Modern Imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Health.AI | Clinical Assistant", page_icon="⚕️", layout="wide", initial_sidebar_state="expanded")

# --- SAFE CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button {
        background-color: #0ea5e9; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; font-weight: 600; transition: all 0.3s ease; width: 100%;
    }
    .stButton>button:hover {
        background-color: #0284c7; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transform: translateY(-1px); color: white;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid rgba(148, 163, 184, 0.2); margin-bottom: 10px;
    }
    .disclaimer {
        font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- SECURE API KEY FETCHING ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 Groq API key not found! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- SIDEBAR SETUP ---
with st.sidebar:
    st.title("⚕️ Health.AI")
    st.markdown("*Clinical Intelligence Dashboard*")
    st.divider()
    
    st.subheader("📊 Patient Data Source")
    data_source = st.radio(
        "Choose input method:", 
        [
            "🧪 Try Demo (Synthetic Patient)", 
            "📂 Upload Medical Records",
            "🌐 Fetch via FHIR API"
        ]
    )
    
    uploaded_files = None
    if data_source == "📂 Upload Medical Records":
        uploaded_files = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# --- FHIR API INTEGRATION ---
def fetch_fhir_patient_data():
    base_url = "http://hapi.fhir.org/baseR4"
    try:
        patient_resp = requests.get(f"{base_url}/Patient?_count=1&_sort=-_lastUpdated", timeout=10)
        patient_resp.raise_for_status()
        patient_data = patient_resp.json()
        if not patient_data.get('entry'): return None
        
        patient = patient_data['entry'][0]['resource']
        patient_id = patient.get('id', 'Unknown')
        name_info = patient.get('name', [{}])[0]
        full_name = f"{name_info.get('given', [''])[0]} {name_info.get('family', '')}"
        
        return f"Patient Name: {full_name}\nID: {patient_id}\n\nNote: This is limited mock data from public FHIR."
    except:
        return None

# --- CORE PROCESSING FUNCTIONS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(data_source, uploaded_files):
    documents = []
    
    if data_source == "🧪 Try Demo (Synthetic Patient)":
        st.info("Loading realistic synthetic data (Synthea) for demonstration...")
        try:
            loader = TextLoader("demo_patient.txt")
            documents.extend(loader.load())
        except Exception as e:
            st.error("Demo file 'demo_patient.txt' not found in repository. Please create it!")
            return None
            
    elif data_source == "🌐 Fetch via FHIR API":
        st.info("🔄 Reaching out to HAPI FHIR Public Server...")
        fhir_text = fetch_fhir_patient_data()
        if fhir_text:
            documents.append(Document(page_content=fhir_text, metadata={"source": "HAPI FHIR API"}))
        else:
            st.error("Could not retrieve patient data. API might be busy.")
            return None
            
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

st.title("🩻 Patient Profile & Clinical Insights")
st.markdown("Interact with medical records using secure, explainable AI retrieval.")

with st.spinner("Processing medical data..."):
    vectorstore = process_documents(data_source, uploaded_files)

if vectorstore:
    st.success("✅ Data successfully processed, vectorized, and ready for analysis.")
    
    tab_summary, tab_chat = st.tabs(["📋 Clinical Summary", "💬 AI Assistant & Citations"])
    
    with tab_summary:
        st.subheader("Comprehensive Profile Overview")
        if st.button("✨ Generate Medical Summary"):
            with st.spinner("Compiling summary..."):
                query = "Provide a structured clinical summary. Include: Patient Demographics, Active Conditions, Medications, and Recent Vitals. Use bullet points."
                docs = vectorstore.similarity_search(query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"Based ONLY on the following medical records:\n{context}\n\nTask: {query}"
                summary_response = llm.invoke(prompt)
                st.info(summary_response.content)

    with tab_chat:
        st.subheader("Query Patient Records")
        user_query = st.text_input("Ask a specific question about the patient's health or history:")
        
        if user_query:
            with st.spinner("Searching records..."):
                retrieved_docs = vectorstore.similarity_search(user_query, k=3)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # STRICT GUARDRAILS ADDED HERE
                final_prompt = f"""You are a highly accurate clinical AI assistant. Use ONLY the following pieces of context to answer the user's question. 
                Do not provide general medical advice. If the answer is not explicitly contained within the context, state clearly: "I cannot answer this based on the provided medical records."
                
                Context: {context_text}
                
                Question: {user_query}
                
                Helpful Answer:"""
                
                result = llm.invoke(final_prompt)
                
                st.markdown("#### 🤖 AI Response")
                st.info(result.content)
                
                st.markdown("#### 📑 Citations & Source Evidence")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Reference Document {i+1}"):
                        source_name = doc.metadata.get('source', 'demo_patient.txt')
                        st.markdown(f"**Source:** `{source_name}`")
                        st.markdown(f"**Exact Excerpt:**\n> {doc.page_content}")
else:
    if data_source == "📂 Upload Medical Records":
        st.info("📂 Please upload PDF or TXT files in the sidebar to begin analysis.")

# --- CLINICAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Disclaimer:</strong> Health.AI is a demonstration prototype designed for educational and portfolio purposes. 
    It is not a certified medical device. The AI-generated outputs should never be used as a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
