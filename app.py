import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Fin-Intelligence NLP", page_icon="📈", layout="wide")

# --- 1. CSS Visibility Fix ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    [data-testid="stMetricLabel"] { color: #31333F !important; font-weight: bold !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #000000 !important; font-size: 24px !important; }
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Model Loading ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("Devda1421/financial-sentiment-distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_map = {0: "Negative 🔴", 1: "Neutral ⚪", 2: "Positive 🟢"}
label_names = ["Negative", "Neutral", "Positive"]

# --- 3. UI Layout ---
st.title("📊 Financial Sentiment Intelligence")
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 News Headline Input")
    headline_input = st.text_area("Enter financial context:", height=180, placeholder="Type news here...")
    analyze_btn = st.button("🔍 Run Signal Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("🎯 Analysis Output")
    if analyze_btn and headline_input:
        inputs = tokenizer(headline_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
        pred_idx = probs.index(max(probs))
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Market Sentiment", label_map[pred_idx])
        m_col2.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")
        
        fig = px.bar(pd.DataFrame({'Sentiment': label_names, 'Prob': probs}), x='Sentiment', y='Prob', color='Sentiment')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Awaiting input for analysis.")
