import streamlit as st
import os
import tempfile
import requests
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Health.AI | Clinical Assistant", page_icon="⚕️", layout="wide")

# --- SIDEBAR & API SETUP ---
with st.sidebar:
    st.title("⚕️ Health.AI")
    st.markdown("**Medical Record Analyzer**")
    st.divider()
    
    groq_api_key = st.text_input("Enter Groq API Key:", type="password", help="Get a free key at console.groq.com")
    st.divider()
    
    st.subheader("Patient Data Source")
    data_source = st.radio(
        "Choose input method:", 
        ["Fetch via FHIR API (Realistic Mock Data)", "Upload Medical Records"]
    )
    
    uploaded_files = None
    if data_source == "Upload Medical Records":
        uploaded_files = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# --- FHIR API INTEGRATION ---
def fetch_fhir_patient_data():
    """Fetches realistic synthetic patient data from the public HAPI FHIR server."""
    base_url = "http://hapi.fhir.org/baseR4"
    patient_text = ""
    
    try:
        # 1. Fetch a random patient
        patient_resp = requests.get(f"{base_url}/Patient?_count=1&_sort=-_lastUpdated", timeout=10)
        patient_resp.raise_for_status()
        patient_data = patient_resp.json()
        
        if not patient_data.get('entry'):
            return None
            
        patient = patient_data['entry'][0]['resource']
        patient_id = patient.get('id', 'Unknown')
        name_info = patient.get('name', [{}])[0]
        full_name = f"{name_info.get('given', [''])[0]} {name_info.get('family', '')}"
        gender = patient.get('gender', 'Unknown')
        birth_date = patient.get('birthDate', 'Unknown')
        
        patient_text += f"Patient Name: {full_name}\nID: {patient_id}\nGender: {gender}\nBirth Date: {birth_date}\n\n"
        
        # 2. Fetch conditions for this patient
        cond_resp = requests.get(f"{base_url}/Condition?patient={patient_id}", timeout=10)
        if cond_resp.status_code == 200:
            cond_data = cond_resp.json()
            patient_text += "Medical Conditions:\n"
            if cond_data.get('entry'):
                for entry in cond_data['entry']:
                    condition = entry['resource'].get('code', {}).get('text', 'Unknown Condition')
                    patient_text += f"- {condition}\n"
            else:
                patient_text += "- No recorded conditions in this dataset.\n"
                
        return patient_text
        
    except Exception as e:
        st.error(f"Failed to fetch data from FHIR API: {e}")
        return None

# --- CORE PROCESSING FUNCTIONS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_documents(data_source, uploaded_files):
    documents = []
    
    if data_source == "Fetch via FHIR API (Realistic Mock Data)":
        st.info("Reaching out to HAPI FHIR Public Server...")
        fhir_text = fetch_fhir_patient_data()
        if fhir_text:
            # Convert text string into a LangChain Document object
            doc = Document(page_content=fhir_text, metadata={"source": "HAPI FHIR Public API"})
            documents.append(doc)
            with st.expander("👀 View Raw Data Fetched from API"):
                st.text(fhir_text)
        else:
            st.error("Could not retrieve patient data. The public API might be busy.")
            return None
            
    elif data_source == "Upload Medical Records" and uploaded_files:
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
if not groq_api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar to activate Health.AI.")
    st.stop()

llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)

st.title("Patient Profile & Insights")

with st.spinner("Processing medical data..."):
    vectorstore = process_documents(data_source, uploaded_files)

if vectorstore:
    st.success("Data successfully processed and vectorized.")
    
    tab_summary, tab_chat = st.tabs(["📋 Patient Summary", "💬 AI Assistant & Citations"])
    
    with tab_summary:
        st.subheader("Comprehensive Profile")
        if st.button("Generate Medical Summary"):
            with st.spinner("Compiling summary..."):
                query = "Provide a structured summary of the patient including: Name, Demographics, and Primary Medical Conditions."
                docs = vectorstore.similarity_search(query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"Based ONLY on the following medical records:\n{context}\n\nTask: {query}"
                summary_response = llm.invoke(prompt)
                st.write(summary_response.content)

    with tab_chat:
        st.subheader("Query Patient Records")
        user_query = st.text_input("Ask a question about the patient's health or history:")
        
        if user_query:
            with st.spinner("Searching records..."):
                prompt_template = """Use the following pieces of context to answer the user's question. 
                If the answer is not contained within the context, state clearly: "I cannot answer this based on the provided medical records."
                
                Context: {context}
                Question: {question}
                
                Helpful Answer:"""
                
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                result = qa_chain.invoke({"query": user_query})
                
                st.markdown("### Response")
                st.info(result['result'])
                
                st.markdown("### Citations & Rationale")
                for i, doc in enumerate(result['source_documents']):
                    with st.expander(f"Source Document Reference {i+1}"):
                        source_name = doc.metadata.get('source', 'Unknown Source')
                        st.markdown(f"**Source:** {source_name}")
                        st.markdown(f"**Excerpt:**\n> {doc.page_content}")
else:
    if data_source == "Upload Medical Records":
        st.info("Please upload PDF or TXT files in the sidebar to begin.")
