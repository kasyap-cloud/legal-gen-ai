import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage, documentai_v1 as documentai, firestore
from google.oauth2 import service_account

# Vertex AI RAG and Generative AI imports
import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

# -------------------------
# Load environment
# -------------------------
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")
SERVICE_ACCOUNT_KEY_PATH = os.getenv("SERVICE_ACCOUNT_KEY_PATH", "service_account_key.json")

# -------------------------
# Regions
# -------------------------
DOC_AI_REGION = "us"         
RAG_REGION = "us-east4"       
# -------------------------
# Credentials
# -------------------------
try:
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_PATH)
except FileNotFoundError:
    st.error(f"Service account key file not found at {SERVICE_ACCOUNT_KEY_PATH}")
    st.stop()

storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)
documentai_client = documentai.DocumentProcessorServiceClient(credentials=credentials)
db = firestore.Client(project=PROJECT_ID, credentials=credentials, database="docs-db")

# -------------------------
# Init Vertex AI
# -------------------------
vertexai.init(project=PROJECT_ID, location=RAG_REGION)

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h1 style='text-align: center;'>Lexi-Gen: AI Legal Assistant</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

with col2:
    st.markdown("<h1 style='text-align: center;'>Ask Questions (RAG Chat)</h1>", unsafe_allow_html=True)
    user_question = st.text_input("Enter your legal query here:")

# -------------------------
# Upload PDF to GCS
# -------------------------
def upload_to_gcs(uploaded_file):
    uploaded_file.seek(0)
    destination = f"uploads/{uploaded_file.name}-{uuid.uuid4()}"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination)
    blob.upload_from_file(uploaded_file, content_type="application/pdf")
    return f"gs://{BUCKET_NAME}/{destination}"

# -------------------------
# Process PDF with Document AI
# -------------------------
def process_document(gcs_uri):
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()

    raw_document = documentai.RawDocument(content=content, mime_type="application/pdf")
    processor_name = documentai_client.processor_path(PROJECT_ID, DOC_AI_REGION, PROCESSOR_ID)
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
    response = documentai_client.process_document(request=request)
    return response.document.text if response.document and response.document.text else ""

# -------------------------
# Save full text to Firestore
# -------------------------
def save_full_text_to_firestore(file_name, gcs_uri, full_text):
    doc_ref = db.collection("contracts").document(str(uuid.uuid4()))
    doc_ref.set({
        "file_name": file_name,
        "source_uri": gcs_uri,
        "full_text": full_text,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "rag_exported_to_gcs": False,
        "rag_gcs_path": None,
    })
    return doc_ref.id

# -------------------------
# Export Firestore → GCS
# -------------------------
def export_firestore_to_gcs(target_prefix="rag_imports/"):
    collection_ref = db.collection("contracts")
    exported_paths = []
    bucket = storage_client.bucket(BUCKET_NAME)

    for doc_snapshot in collection_ref.where("rag_exported_to_gcs", "==", False).stream():
        doc = doc_snapshot.to_dict()
        full_text = doc.get("full_text", "")
        if not full_text:
            continue

        doc_id = doc_snapshot.id
        blob_name = f"{target_prefix}{doc_id}.txt"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(full_text, content_type="text/plain; charset=utf-8")
        gcs_path = f"gs://{BUCKET_NAME}/{blob_name}"
        exported_paths.append(gcs_path)

        doc_snapshot.reference.update({
            "rag_exported_to_gcs": True,
            "rag_gcs_path": gcs_path,
            "rag_exported_at": firestore.SERVER_TIMESTAMP
        })

    return exported_paths

# -------------------------
# Setup Corpus (new API)
# -------------------------
def setup_rag_corpus_from_gcs(gcs_paths):
    corpus = rag.create_corpus(display_name=f"legal_corpus_{uuid.uuid4()}")
    rag.import_files(corpus_name=corpus.name, paths=gcs_paths)
    return corpus


def wait_for_indexing(corpus, timeout=120):
    st.info("Skipping explicit indexing check (assuming ingestion will complete).")
    return True

# -------------------------
# Query RAG Corpus and Generate Answer with Gemini
# -------------------------
def query_rag_with_llm(corpus, question):
    """
    Performs a RAG query and uses a Gemini model to generate a synthesized answer.
    """
    try:
        # Create a Retrieval Tool from the RAG Corpus
        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
                    # similarity_top_k=3
                ),
            )
        )
        # Initialize a Gemini model with the RAG tool
        rag_model = GenerativeModel("gemini-2.5-flash-lite", tools=[rag_retrieval_tool])

        prompt = f"Using the provided legal documents, answer the following question: {question}"

        response = rag_model.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            return "Failed to generate an answer. No text found in response."
    
    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
        return "Failed to generate an answer."

# -------------------------
# Process uploaded file
# -------------------------
if uploaded_file:
    st.info("Uploading PDF to GCS...")
    gcs_uri = upload_to_gcs(uploaded_file)
    st.success(f"Uploaded to {gcs_uri}")

    st.info("Processing with Document AI...")
    full_text = process_document(gcs_uri)
    if not full_text:
        st.error("Document AI returned empty text.")
        st.stop()

    st.subheader("Preview (first 500 chars)")
    st.write(full_text[:500] + "...")

    st.info("Saving full text to Firestore...")
    doc_id = save_full_text_to_firestore(uploaded_file.name, gcs_uri, full_text)
    st.success(f"Saved in Firestore contracts/{doc_id}")

    st.info("Exporting Firestore docs to GCS (TXT)...")
    exported_paths = export_firestore_to_gcs()
    if not exported_paths:
        st.error("No text exported to GCS.")
        st.stop()

    st.info("Creating RAG Corpus and ingesting TXT files...")
    corpus = setup_rag_corpus_from_gcs(exported_paths)
    st.success(f"Corpus created: {corpus.name}")
    st.session_state["corpus"] = corpus
    st.session_state["corpus_ready"] = wait_for_indexing(corpus)

# -------------------------
# RAG Chat
# -------------------------
if user_question:
    if "corpus" not in st.session_state or not st.session_state.get("corpus_ready"):
        st.warning("Corpus is not ready. Please upload a file and wait for indexing to complete.")
    else:
        corpus = st.session_state["corpus"]
        answer = query_rag_with_llm(corpus, user_question)
        col2.markdown(f"**Answer:**\n{answer}")
