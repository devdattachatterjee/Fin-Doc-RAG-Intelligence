import streamlit as st
from openai import OpenAI
import pypdf
import faiss
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")
st.markdown("""<style>.stApp { background-color: #0e1117; color: #ffffff; }</style>""", unsafe_allow_html=True)

# --- 2. Pure Python Text Chunker ---
def get_text_chunks(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- 3. Authentication ---
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    try:
        # Initialize official OpenAI Client
        client = OpenAI(api_key=api_key)

        # --- 4. Document Processing (Vanilla Python) ---
        st.sidebar.header("📁 Data Ingestion")
        uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type="pdf")
        
        if uploaded_file:
            if "vector_index" not in st.session_state:
                with st.status("🧠 Processing document from scratch...", expanded=True) as status:
                    # 4a. Read PDF
                    st.write("Extracting raw text...")
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    raw_text = ""
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text() + "\n"
                    
                    # 4b. Chunk Text
                    st.write("Chunking text manually...")
                    chunks = get_text_chunks(raw_text)
                    st.session_state.chunks = chunks
                    
                    # 4c. Generate Embeddings (OpenAI API)
                    st.write("Generating vector embeddings...")
                    response = client.embeddings.create(input=chunks, model="text-embedding-3-small")
                    embeddings = [data.embedding for data in response.data]
                    embedding_matrix = np.array(embeddings, dtype=np.float32)
                    
                    # 4d. Build FAISS Index
                    st.write("Building FAISS index...")
                    dimension = embedding_matrix.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(embedding_matrix)
                    st.session_state.vector_index = index
                    
                    status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)

        # --- 5. Chat Interface ---
        st.title("🤖 Financial Signal Assistant")
        st.caption("Custom-Built RAG Architecture (Zero Framework Dependencies)")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about the uploaded file..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if "vector_index" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving facts..."):
                        
                        # Step A: Embed the user's question
                        query_res = client.embeddings.create(input=[prompt], model="text-embedding-3-small")
                        query_embed = np.array([query_res.data[0].embedding], dtype=np.float32)
                        
                        # Step B: Search the FAISS index for the top 3 closest chunks
                        distances, indices = st.session_state.vector_index.search(query_embed, k=3)
                        retrieved_chunks = [st.session_state.chunks[i] for i in indices[0]]
                        context_string = "\n\n---\n\n".join(retrieved_chunks)
                        
                        # Step C: Send Context + Question to GPT-4o-mini
                        system_prompt = f"You are a helpful financial assistant. Answer the user's question strictly based on the following context:\n\n{context_string}"
                        
                        llm_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0
                        )
                        
                        answer = llm_response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Please upload a PDF in the sidebar first!")
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("🔑 Please enter your OpenAI API Key in the sidebar or add it to Streamlit Secrets to begin.")
