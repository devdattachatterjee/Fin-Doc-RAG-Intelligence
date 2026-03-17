import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- Page Config & UI Styling ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")

# Custom CSS for a dark, professional "Banking Terminal" look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetric"] { background-color: #1e2130; border-radius: 10px; padding: 15px; border: 1px solid #3e4251; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 1. Load Resources (Cached for Speed) ---
@st.cache_resource
def get_tools():
    # Only use secrets if they exist, otherwise app will prompt in sidebar
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAIEmbeddings(openai_api_key=api_key), ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
    except:
        return None, None

embeddings, llm = get_tools()

# --- 2. Sidebar: Document Management ---
with st.sidebar:
    st.title("🏦 Fin-Doc Control")
    st.markdown("---")
    uploaded_file = st.file_uploader("Ingest Financial PDF", type="pdf")
    
    if not embeddings:
        st.warning("⚠️ OpenAI API Key not found in Secrets.")
        user_key = st.text_input("Enter API Key manually:", type="password")
        if user_key:
            embeddings = OpenAIEmbeddings(openai_api_key=user_key)
            llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=user_key)

# --- 3. Knowledge Base Ingestion ---
if uploaded_file and embeddings:
    if "vector_db" not in st.session_state:
        with st.status("🏗️ Building Knowledge Base...", expanded=True) as status:
            # Save and load
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write("📄 Parsing Document...")
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            
            st.write("✂️ Splitting for Contextual Accuracy...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = splitter.split_documents(docs)
            
            st.write("🧠 Generating Semantic Embeddings...")
            st.session_state.vector_db = FAISS.from_documents(texts, embeddings)
            status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)

# --- 4. Conversational Interface ---
st.title("🤖 Financial Signal Assistant")
st.caption("Context-Aware Analysis of Regulatory and Corporate Filings")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Query the document (e.g., 'Analyze the GNPA trends')"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving facts..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                response = qa_chain({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                st.markdown(answer)
                
                # Professional Source Citation Layout
                with st.expander("📍 View Source Evidence"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')})**")
                        st.caption(doc.page_content[:300] + "...")

                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload a document first to activate the AI assistant.")
