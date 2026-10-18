import tempfile
import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vetorial_db(pdf_bytes) -> Chroma:
    """
    Cria um banco de dados vetorial a partir de um arquivo PDF em bytes e retorna o objeto Chroma correspondente.
    """
    
    # Cria um arquivo temporário e salva o conteúdo do PDF nele
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150) 
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/msmarco-bert-base-dot-v5")
    
    vectordb = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory = st.session_state.chroma_dir)

    return vectordb