import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# The Modern LangChain Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Page Config ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")

# --- 2. Authentication ---
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key, temperature=0)

        # --- 3. Sidebar Upload ---
        uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type="pdf")
        
        if uploaded_file:
            if "vector_db" not in st.session_state:
                with st.spinner("Processing document..."):
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader("temp.pdf")
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    texts = splitter.split_documents(docs)
                    st.session_state.vector_db = FAISS.from_documents(texts, embeddings)
                    st.success("✅ Knowledge Base Ready!")

        # --- 4. Chat Interface ---
        st.title("🤖 Financial Signal Assistant")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if "vector_db" in st.session_state:
                with st.chat_message("assistant"):
                    
                    # Modern LangChain Prompt & Chain Setup
                    system_prompt = (
                        "You are a financial analysis assistant. Use the retrieved context to answer the question. "
                        "If you don't know the answer, say that you don't know. Keep your answers factual.\n\n"
                        "{context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    qa_chain = create_retrieval_chain(st.session_state.vector_db.as_retriever(), question_answer_chain)
                    
                    # Execute the chain
                    response = qa_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Please upload a PDF first!")
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("🔑 Please enter your OpenAI API Key in the sidebar or add it to Streamlit Secrets.")
