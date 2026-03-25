import streamlit as st
import os
import tempfile

# Bulletproof Modern Imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
# Mobile optimization: let Streamlit handle sidebar state automatically
st.set_page_config(page_title="AI-Health Assistant", page_icon="⚕️", layout="wide")

# --- MEDICAL GREEN CUSTOM CSS (Mobile Optimized) ---
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

# --- SYNTHETIC PATIENT DATABASE ---
DEMO_PATIENTS = {
    "Evelyn (Cardiology & Diabetes)": """SYNTHETIC PATIENT RECORD - ID: SYN-8472910
Name: Evelyn Carter | DOB: 1945-08-22 | Gender: Female
CHIEF COMPLAINT: Routine 6-month follow-up. Reports occasional mild dizziness when standing.
ACTIVE CONDITIONS: Essential Hypertension (2012), Type 2 Diabetes Mellitus (2015), Osteoarthritis of right knee (2019).
MEDICATIONS: Lisinopril 20mg daily, Metformin 1000mg BID, Ibuprofen 400mg PRN.
VITALS (2024-02-15): BP: 138/88 mmHg, HR: 72 bpm, Weight: 165 lbs, HbA1c: 7.2%, Fasting Glucose: 130 mg/dL.
PLAN: Continue Lisinopril. Diabetes well-controlled. Counselled on orthostatic hypotension. Hydration advised.""",
    
    "Marcus (Asthma Profile)": """SYNTHETIC PATIENT RECORD - ID: SYN-3920114
Name: Marcus Thorne | DOB: 1982-11-04 | Gender: Male
CHIEF COMPLAINT: Presents with persistent dry cough and shortness of breath exacerbated by exercise and cold air. 
ACTIVE CONDITIONS: Moderate Persistent Asthma (Diagnosed 1995), Seasonal Allergic Rhinitis.
MEDICATIONS: Fluticasone/Salmeterol 250/50 mcg inhaler BID, Albuterol 90mcg PRN, Cetirizine 10mg daily.
VITALS (2024-03-01): BP: 120/78 mmHg, HR: 84 bpm, SpO2: 96% on room air.
EXAM: Expiratory wheezing noted in bilateral lower lung fields. 
PLAN: Step up asthma therapy. Prescribed oral Prednisone 40mg taper for 5 days. Follow up in 2 weeks.""",

    "Sarah (Post-Op Recovery)": """SYNTHETIC PATIENT RECORD - ID: SYN-5592833
Name: Sarah Jenkins | DOB: 1968-04-19 | Gender: Female
CHIEF COMPLAINT: 2-week post-operative check for Left Total Hip Arthroplasty (THA).
ACTIVE CONDITIONS: Left Hip Osteoarthritis (Surgical), Hyperlipidemia.
MEDICATIONS: Atorvastatin 20mg daily, Apixaban 2.5mg BID (DVT prophylaxis), Acetaminophen 1000mg TID.
VITALS (2024-03-10): BP: 128/82 mmHg, HR: 76 bpm, Temp: 98.6F.
EXAM: Surgical incision site clean, dry, intact. No erythema or purulent drainage. Mild edema in left lower extremity. Range of motion improving.
PLAN: Discontinue narcotic pain relievers. Continue Apixaban for 2 more weeks. Begin outpatient physical therapy twice weekly."""
}

# --- SIDEBAR: INFORMATION & ARCHITECTURE ---
with st.sidebar:
    st.markdown("<h2 style='color: #059669;'>ℹ️ About the AI</h2>", unsafe_allow_html=True)
    st.markdown("This menu contains technical details about how the AI processes your data.")
    st.divider()
    
    with st.expander("⚙️ AI Architecture Diagram"):
        st.markdown("**How the Data Flows**")
        st.graphviz_chart('''
            digraph {
                bgcolor="transparent"
                node [shape=box, style=filled, fillcolor="#ecfdf5", color="#059669", fontname="Helvetica", fontsize=10]
                edge [color="#047857", arrowsize=0.7]
                A [label="1. Patient Data\n(Uploaded/Demo)"]
                B [label="2. Text Processing\n(Chunking)"]
                C [label="3. AI Vector Database\n(FAISS)"]
                D [label="4. AI Brain\n(Groq LLaMA-3.1)"]
                E [label="5. Final Summary\n& Answers"]
                A -> B -> C
                C -> D [label=" Secure Search", fontsize=9, fontcolor="#059669"]
                D -> E
            }
        ''')
    
    with st.expander("🛠️ Tech Stack Used"):
        st.markdown("""
        * **AI Model:** Groq (LLaMA-3.1-8B-Instant)
        * **Memory:** FAISS Vector Database
        * **Logic:** Semantic Search mapping
        * **Safety:** Strict medical guardrails embedded in AI prompts.
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

# --- MAIN PAGE: MOBILE FRIENDLY UI ---
st.title("⚕️ AI-Health Assistant")
st.markdown("Easily analyze medical records, generate clinical summaries, and ask the AI questions safely.")

# Front-and-Center Controls (No longer hidden in sidebar)
st.markdown("### 1️⃣ Select Patient Data")
data_source = st.radio(
    "How would you like to load data?", 
    ["🧪 Try AI-Demo Patients", "📂 Upload Patient Records"],
    horizontal=True # Mobile friendly horizontal layout
)

selected_patient_data = None
uploaded_files = None

# Show specific inputs based on radio selection
if data_source == "🧪 Try AI-Demo Patients":
    patient_name = st.selectbox("Choose a demo patient profile:", list(DEMO_PATIENTS.keys()))
    selected_patient_data = DEMO_PATIENTS[patient_name]
else:
    uploaded_files = st.file_uploader("Securely upload EHR Data (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

st.divider()

# Process data and run the main AI interface
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

with st.spinner("AI is analyzing the records..."):
    vectorstore = process_documents(data_source, selected_patient_data, uploaded_files)

if vectorstore:
    st.success("✅ Patient Data Loaded. The AI Assistant is ready.")
    
    tab_summary, tab_chat = st.tabs(["📋 AI Clinical Summary", "💬 Ask the AI Assistant"])
    
    with tab_summary:
        st.markdown("### Auto-Generated Patient Breakdown")
        if st.button("✨ Generate AI Summary"):
            with st.spinner("AI is writing the summary..."):
                query = "Provide a highly detailed, structured clinical summary. Include sections for: Patient Demographics, Chief Complaint, Active Conditions, Medications, Vitals, and Plan. Use clear bullet points and bold headers."
                docs = vectorstore.similarity_search(query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"Based ONLY on the following medical records:\n{context}\n\nTask: {query}"
                summary_response = llm.invoke(prompt)
                
                st.info(summary_response.content)
                
                st.download_button(
                    label="📥 Download AI Summary (.txt)",
                    data=summary_response.content,
                    file_name="AI_Clinical_Summary.txt",
                    mime="text/plain"
                )

    with tab_chat:
        st.markdown("### Ask Questions About the Records")
        user_query = st.text_input("Type a question (e.g., 'What are the patient's current medications?'):")
        
        if user_query:
            with st.spinner("AI is searching the documents..."):
                retrieved_docs = vectorstore.similarity_search(user_query, k=3)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                final_prompt = f"""You are a helpful, professional clinical AI assistant. 
                Use ONLY the following pieces of context to answer the user's question. Provide a detailed, clear response.
                Do not provide medical advice outside of the context. If the answer is not in the text, say: "I cannot answer this based on the provided records."
                
                Context: {context_text}
                
                Question: {user_query}
                
                Answer:"""
                
                result = llm.invoke(final_prompt)
                
                st.markdown("#### 🤖 AI Response")
                st.success(result.content)
                
                st.markdown("#### 📑 Where the AI found this:")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Source Document {i+1}"):
                        source_name = doc.metadata.get('source', 'AI Demo Database')
                        st.markdown(f"**Source:** `{source_name}`")
                        st.markdown(f"**Exact Text:**\n> {doc.page_content}")
else:
    if data_source == "📂 Upload Patient Records":
        st.info("📂 Please upload files above to activate the AI.")

# --- CLINICAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Regulatory Disclaimer:</strong> This AI-Health Assistant is a demonstration prototype. 
    It is not certified medical software. AI-generated outputs should never replace professional medical diagnosis or treatment.
</div>
""", unsafe_allow_html=True)
