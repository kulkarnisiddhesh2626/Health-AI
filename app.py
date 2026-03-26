import streamlit as st
import os
import tempfile
from datetime import datetime
from PIL import Image
import pytesseract

# Bulletproof Modern Imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI-Health Assistant", page_icon="⚕️", layout="wide")

# --- CLEAN MEDICAL UI CSS ---
st.markdown("""
<style>
    .stButton>button {
        background-color: #059669; color: white; border-radius: 6px; border: none; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #047857; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: white;
    }
    div[data-testid="stExpander"] {
        border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 8px;
    }
    .disclaimer {
        font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
    }
    div.row-widget.stRadio > div { flex-direction: row; flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)

# --- SECURE API KEY FETCHING ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 Groq API key not found! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- 10 EXPANDED SYNTHETIC PATIENTS ---
DEMO_PATIENTS = {
    "Evelyn Carter (Type 2 Diabetes)": """Date: 2024-02-15. Patient: Evelyn Carter. DOB: 1945-08-22. Gender: Female.
2012-04-10: Diagnosed with Essential Hypertension. Prescribed Lisinopril 20mg.
2015-09-22: Diagnosed with Type 2 Diabetes. Started Metformin 500mg.
2023-08-14: Metformin increased to 1000mg BID due to elevated HbA1c (8.1%).
2024-02-15 (Current Visit): Routine 6-month follow-up. BP: 138/88. HbA1c improved to 7.2%. Plan: Continue meds, stable.""",

    "Marcus Thorne (Asthma Exacerbation)": """Date: 2024-03-01. Patient: Marcus Thorne. DOB: 1982-11-04. Gender: Male.
1995-06-15: Diagnosed with Moderate Persistent Asthma.
2024-02-28: Patient visited Urgent Care for dry cough and severe shortness of breath. SpO2 was 92%. Given Albuterol nebulizer.
2024-03-01 (Current Visit): Follow-up. SpO2 96%. Expiratory wheezing present. Plan: Prescribed Prednisone 40mg taper for 5 days.""",

    "Robert Vance (Acute Stroke Protocol)": """Date: 2024-03-20. Patient: Robert Vance. DOB: 1955-02-14. Gender: Male.
2018-11-05: Diagnosed with Atrial Fibrillation. Non-compliant with Eliquis.
2024-03-20 14:15: Arrived at ER. Sudden onset left-sided weakness and slurred speech starting at 13:30.
2024-03-20 14:30: Vitals - BP: 195/110, HR: 115. Pronator drift observed.
2024-03-20 14:45: STAT CT Head ordered. Neurology consulted for tPA administration.""",

    "Sarah Jenkins (Post-Op Orthopedics)": """Date: 2024-03-10. Patient: Sarah Jenkins. DOB: 1968-04-19. Gender: Female.
2024-02-25: Underwent Left Total Hip Arthroplasty (THA). Uncomplicated.
2024-02-28: Discharged home with Apixaban 2.5mg BID and Oxycodone.
2024-03-10 (Current Visit): 2-week post-op check. Incision clean, no erythema. Plan: Discontinue Oxycodone, start outpatient PT.""",

    "David Kim (Chronic Kidney Disease)": """Date: 2024-01-12. Patient: David Kim. DOB: 1960-07-30. Gender: Male.
2019-03-10: Diagnosed with Stage 2 CKD and Hypertension.
2023-12-05: Lab results showed eGFR declined to 42 mL/min (Stage 3 CKD).
2024-01-12 (Current Visit): BP 145/90. Complains of mild fatigue. Plan: Referral to Nephrology. Switch from Amlodipine to Losartan 50mg for renal protection.""",

    "Maria Gonzalez (Rheumatoid Arthritis)": """Date: 2024-02-20. Patient: Maria Gonzalez. DOB: 1975-12-05. Gender: Female.
2010-08-14: Diagnosed with Rheumatoid Arthritis. Maintained on Methotrexate.
2024-02-10: Patient called clinic reporting severe joint stiffness in hands lasting >2 hours every morning.
2024-02-20 (Current Visit): Swelling in bilateral MCP joints. Plan: Flare-up indicated. Adding short course Methylprednisolone and bridging to Adalimumab.""",

    "James O'Connor (Post-MI Cardiology)": """Date: 2024-03-22. Patient: James O'Connor. DOB: 1958-09-12. Gender: Male.
2024-01-15: Suffered Acute STEMI. Underwent PCI with 2 stents placed in LAD.
2024-01-18: Discharged on Dual Antiplatelet Therapy (Aspirin + Ticagrelor), Atorvastatin 80mg, and Metoprolol.
2024-03-22 (Current Visit): Cardiac rehab evaluation. No chest pain. Echo shows LVEF at 50%. Plan: Continue current regimen, patient doing well.""",

    "Anita Patel (Hypothyroidism)": """Date: 2024-02-28. Patient: Anita Patel. DOB: 1988-04-25. Gender: Female.
2021-05-10: Diagnosed with Hashimoto's Thyroiditis. Started Levothyroxine 75mcg.
2024-02-15: Routine labs - TSH elevated at 6.5 mIU/L. Patient reports weight gain and cold intolerance.
2024-02-28 (Current Visit): Plan: Increase Levothyroxine to 88mcg daily. Recheck TSH in 6 weeks.""",

    "Lucas Wright (Pediatric Type 1 Diabetes)": """Date: 2024-03-05. Patient: Lucas Wright. DOB: 2014-11-10. Gender: Male.
2024-03-01: Brought to ER by parents for extreme thirst, frequent urination, and weight loss. Blood glucose 450 mg/dL. Diagnosed with Type 1 Diabetes.
2024-03-05 (Current Visit): Pediatric Endocrinology follow-up. Parents trained on insulin glargine and lispro administration. Plan: Close monitoring of blood sugars.""",

    "Emily Chen (Obstetrics - Gestational DM)": """Date: 2024-03-18. Patient: Emily Chen. DOB: 1992-02-14. Gender: Female.
2023-10-05: Confirmed intrauterine pregnancy. 
2024-03-10: 24-week Oral Glucose Tolerance Test (OGTT) failed (1-hour glucose 185 mg/dL).
2024-03-18 (Current Visit): 25 weeks gestation. Diagnosed with Gestational Diabetes. Plan: Dietary counseling initiated. Patient to check blood glucose 4x daily."""
}

# --- INITIALIZE SESSION STATE FOR CLICKABLE FAQs ---
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

def set_query(query_text):
    st.session_state.user_query = query_text

# --- SIDEBAR: CLEAN & STRUCTURED ---
with st.sidebar:
    st.markdown("<h2 style='color: #059669;'>⚙️ System Overview</h2>", unsafe_allow_html=True)
    
    with st.expander("📊 AI Architecture & Flow", expanded=True):
        st.markdown("**Semantic Retrieval-Augmented Generation (RAG)**")
        st.graphviz_chart('''
            digraph {
                bgcolor="transparent"
                node [shape=box, style=filled, fillcolor="#ecfdf5", color="#059669", fontname="Helvetica", fontsize=9]
                edge [color="#047857", arrowsize=0.6]
                A [label="Patient Records\n(Images/Docs)"]
                B [label="OCR & Text Chunking"]
                C [label="FAISS Vector DB"]
                D [label="Groq LLaMA AI"]
                E [label="Clinical Output"]
                A -> B -> C -> D -> E
            }
        ''')
        st.markdown("*Multi-modal data is read, vectorized into mathematical embeddings, and passed to the LLM for high-accuracy retrieval.*")
        
    with st.expander("🛠️ Tech Stack & Processing", expanded=False):
        st.markdown("""
        * **LLM Engine:** Groq (LLaMA-3.1-8B-Instant)
        * **Vector DB:** FAISS (Facebook AI Similarity Search)
        * **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
        * **Ingestion:** Supports PDF, TXT, DOCX, CSV, and **Images (PNG, JPG, JPEG via Tesseract OCR)**.
        """)
        
    with st.expander("📋 Scope & Limitations", expanded=False):
        st.markdown("""
        * **Scope:** Instantly summarizes chronologies, extracts vitals, and highlights clinical events from text and scanned images.
        * **Safety:** Strictly uses provided context. Will not generate outside medical advice.
        * **Limitations:** Not a diagnostic tool. Must be reviewed by a certified physician.
        """)

# --- HELPER FUNCTION: COLOR PARSER ---
def render_triage_response(raw_text, process_time):
    st.caption(f"⏱️ **Processed Timeline:** {process_time}")
    if "[CRITICAL]" in raw_text:
        st.error(f"🔴 **CRITICAL STATUS**\n\n{raw_text.replace('[CRITICAL]', '').strip()}")
    elif "[MODERATE]" in raw_text:
        st.warning(f"🟡 **MODERATE STATUS**\n\n{raw_text.replace('[MODERATE]', '').strip()}")
    elif "[STABLE]" in raw_text:
        st.success(f"🟢 **STABLE STATUS**\n\n{raw_text.replace('[STABLE]', '').strip()}")
    else:
        st.info(raw_text)

# --- CORE PROCESSING FUNCTIONS (NOW WITH OCR) ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(data_source, selected_data, uploaded_files):
    documents = []
    
    if data_source == "🧪 Load Demo Patient" and selected_data:
        documents.append(Document(page_content=selected_data, metadata={"source": "Database"}))
            
    elif data_source == "📂 Upload Medical Records" and uploaded_files:
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
                elif file_extension in ["png", "jpg", "jpeg"]:
                    # OCR Image Processing
                    extracted_text = pytesseract.image_to_string(Image.open(temp_file_path))
                    documents.append(Document(page_content=extracted_text, metadata={"source": file.name}))
            except Exception as e:
                st.error(f"Error processing {file.name}. Ensure Tesseract OCR is installed on the server. Error: {e}")
            finally:
                os.unlink(temp_file_path)
                
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = load_embeddings()
    return FAISS.from_documents(texts, embeddings)

# --- MAIN PAGE: UI ---
st.title("⚕️ AI-Health Assistant")
st.markdown("Clinical intelligence and semantic EHR analysis across Text, Docs, and Images.")

st.markdown("### 1️⃣ Patient Selection")
data_source = st.radio("Data Source:", ["🧪 Load Demo Patient", "📂 Upload Medical Records"], horizontal=True, label_visibility="collapsed")

selected_patient_data = None
uploaded_files = None

if data_source == "🧪 Load Demo Patient":
    patient_name = st.selectbox("Select Patient Profile:", list(DEMO_PATIENTS.keys()))
    selected_patient_data = DEMO_PATIENTS[patient_name]
else:
    # Uploader now accepts images
    uploaded_files = st.file_uploader(
        "Upload Records (PDF, TXT, DOCX, CSV, PNG, JPG)", 
        type=["pdf", "txt", "docx", "csv", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

st.divider()

llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

with st.spinner("Processing medical context..."):
    vectorstore = process_documents(data_source, selected_patient_data, uploaded_files)

if vectorstore:
    tab_summary, tab_chat = st.tabs(["📋 Auto-Summary & Timeline", "💬 Ask AI Assistant"])
    
    with tab_summary:
        if st.button("✨ Generate Clinical Summary & Timeline"):
            with st.spinner("Analyzing timeline..."):
                query = "Provide a structured summary. Include: 1. Demographics. 2. Chief Complaint. 3. Chronological Timeline of Medical Events (use exact dates). 4. Plan."
                docs = vectorstore.similarity_search(query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"""Based ONLY on the context below, perform the task. 
                First, assess the patient's stability and start your response with exactly ONE tag: [CRITICAL], [MODERATE], or [STABLE].
                Then provide the structured summary.
                Context: {context}\nTask: {query}"""
                
                summary_response = llm.invoke(prompt)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                render_triage_response(summary_response.content, current_time)

    with tab_chat:
        st.markdown("### Quick Queries")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("📋 Summarize the medication plan", on_click=set_query, args=("Summarize the medication plan.",))
        with col2:
            st.button("🫀 Are there abnormal vitals?", on_click=set_query, args=("Are there any abnormal vitals?",))
        with col3:
            st.button("📅 List recent clinical events", on_click=set_query, args=("List the most recent clinical events chronologically.",))

        user_query = st.text_input("Or type a specific question:", value=st.session_state.user_query)
        
        if user_query:
            with st.spinner("Searching records..."):
                retrieved_docs = vectorstore.similarity_search(user_query, k=4)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                final_prompt = f"""Use ONLY the context below. First assess severity and start with exactly ONE tag: [CRITICAL], [MODERATE], or [STABLE]. 
                Then answer the question clearly.
                Context: {context_text}
                Question: {user_query}"""
                
                result = llm.invoke(final_prompt)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                render_triage_response(result.content, current_time)
                
                with st.expander("📑 View Source Evidence"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Excerpt {i+1}:** > {doc.page_content}")
else:
    if data_source == "📂 Upload Medical Records":
        st.info("Please upload files to proceed. Images will automatically be processed using OCR.")

# --- CLINICAL DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Disclaimer:</strong> Demonstration prototype only. Not for medical diagnosis.
</div>
""", unsafe_allow_html=True)
