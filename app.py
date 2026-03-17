import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. Setup & CSS ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Authentication ---
# Using .get() to prevent crashing if the secret isn't there yet
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)

        # --- 3. Sidebar: Document Processing ---
        st.sidebar.header("📁 Data Ingestion")
        uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type="pdf")
        
        if uploaded_file:
            if "vector_db" not in st.session_state:
                with st.status("🧠 Processing document for RAG...", expanded=True) as status:
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader("temp.pdf")
                    docs = loader.load()
                    
                    # Splitting text to fit LLM context windows
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    texts = splitter.split_documents(docs)
                    
                    # Creating the searchable vector database
                    st.session_state.vector_db = FAISS.from_documents(texts, embeddings)
                    status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)

        # --- 4. Chat Interface ---
        st.title("🤖 Financial Signal Assistant")
        st.caption("Factual analysis of banking reports using RAG architecture")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User input
        if prompt := st.chat_input("Ask a question about the uploaded file..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if "vector_db" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving facts..."):
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm, 
                            chain_type="stuff", 
                            retriever=st.session_state.vector_db.as_retriever()
                        )
                        # Updated .invoke() syntax for LangChain 2026 compatibility
                        response = qa_chain.invoke({"query": prompt})
                        answer = response["result"]
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Please upload a PDF in the sidebar first!")
                
    except Exception as e:
        st.error(f"Configuration Error: {e}")
else:
    st.info("🔑 Please enter your OpenAI API Key in the sidebar or add it to Streamlit Secrets to begin.")
