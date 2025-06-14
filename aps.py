import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
import os

# ── Atur direktori lokal untuk data NLTK ───────────────────────────────────────
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Download tokenizer 'punkt' ke folder lokal jika belum ada
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)
# ──────────────────────────────────────────────────────────────────────────────

# Konfigurasi halaman
st.set_page_config(
    page_title="AI vs Human Text Detector", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        margin: 20px 0;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .ai-highlight {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .human-highlight {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .neutral-highlight {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .history-card {
        background: white;
        padding: 20px;
        margin: 15px 0;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load model dan tokenizer dengan caching untuk performa
@st.cache_resource
def load_model():
    """Load tokenizer dan model dengan caching untuk performa optimal"""
    model_name = "Alsyabel/AIdetect"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=false)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_text(text, tokenizer, model):
    """Prediksi skor AI vs Manusia untuk teks lengkap"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs[0].tolist()

def simple_sent_tokenize(text):
    return [s.strip() for s in text.split('.') if s.strip()]

def analyze_sentences(text, tokenizer, model):
    """Analisis detailkan per kalimat untuk highlight AI-like content"""
    sentences = simple_sent_tokenize(text)
    results = []
    for sentence in sentences:
        if len(sentence) > 10:
            probs = predict_text(sentence, tokenizer, model)
            results.append({
                'sentence': sentence,
                'ai_prob': probs[1],
                'human_prob': probs[0]
            })
        else:
            results.append({
                'sentence': sentence,
                'ai_prob': 0.5,
                'human_prob': 0.5
            })
    return results

def create_visualization(probs):
    """Buat visualisasi pie chart dan bar chart"""
    
    # Data untuk visualisasi
    labels = ['Human', 'AI']
    values = [probs[0] * 100, probs[1] * 100]
    colors = ['#4CAF50', '#F44336']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie Chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=14
        )])
        fig_pie.update_layout(
            title="Distribution of AI vs Human Score",
            title_x=0.5,
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar Chart
        fig_bar = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f'{v:.1f}%' for v in values],
                textposition='auto',
            )
        ])
        fig_bar.update_layout(
            title="AI vs Human Detection Score",
            title_x=0.5,
            xaxis_title="Category",
            yaxis_title="Percentage (%)",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def extract_text_from_file(uploaded_file):
    """Extract teks dari file yang diupload"""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8").strip()
        else:
            st.error("File type not supported!")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

# Inisialisasi session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🧠 AI vs Human Text Detector</h1>
    <p>Advanced text analysis to detect AI-generated content with detailed insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk informasi dan statistik
with st.sidebar:
    st.markdown("### ℹ️ Information")
    st.info("""
    **Model**: Alsyabel/AIdetect
    
    **Features**:
    • Text & File input support
    • Sentence-level analysis
    • Interactive visualizations
    • History tracking
    
    **Supported Files**: PDF, TXT
    """)
    
    if st.session_state.history:
        st.markdown("### 📊 Statistics")
        total_checks = len(st.session_state.history)
        ai_dominant = sum(1 for _, _, probs, _ in st.session_state.history if probs[1] > 0.5)
        
        st.metric("Total Checks", total_checks)
        st.metric("AI Dominant", f"{ai_dominant}/{total_checks}")
        st.metric("Human Dominant", f"{total_checks - ai_dominant}/{total_checks}")

# Load model
tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Input section
st.markdown("## 📝 Input Section")

# Input type selection
input_type = st.radio(
    "Choose input method:",
    ["✍️ Manual Text Input", "📁 Upload File (.pdf, .txt)"],
    horizontal=True
)

# Title input untuk riwayat
title_input = st.text_input(
    "📋 Title for this analysis (optional):",
    placeholder="e.g., Essay Analysis, Blog Post Check, etc.",
    help="Give a meaningful title to easily identify this analysis in history"
)

text_input = ""

if input_type == "✍️ Manual Text Input":
    text_input = st.text_area(
        "Enter your text here:",
        height=200,
        placeholder="Paste or type the text you want to analyze...",
        help="Enter the text you want to check for AI vs Human authorship"
    )
else:
    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT file for analysis"
    )
    if uploaded_file:
        with st.spinner("Extracting text from file..."):
            text_input = extract_text_from_file(uploaded_file)
        
        if text_input:
            with st.expander("📄 Extracted Text Preview"):
                st.text_area(
                    "Preview (first 500 characters):",
                    value=text_input[:500] + ("..." if len(text_input) > 500 else ""),
                    height=150,
                    disabled=True
                )

# Validasi panjang teks
if text_input and len(text_input) > 2500:
    st.warning("⚠️ Text is too long. Maximum 2500 characters allowed for optimal performance.")
    st.info(f"Current length: {len(text_input)} characters")
    text_input = ""

# Tombol analisis
if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Progress bar untuk user experience
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Analisis utama
            status_text.text("Analyzing overall text...")
            progress_bar.progress(25)
            
            overall_probs = predict_text(text_input, tokenizer, model)
            
            # Analisis per kalimat
            status_text.text("Analyzing individual sentences...")
            progress_bar.progress(50)
            
            sentence_results = analyze_sentences(text_input, tokenizer, model)
            
            progress_bar.progress(75)
            status_text.text("Finalizing results...")
            
            # Simpan hasil
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            title = title_input.strip() or f"Analysis {timestamp}"
            
            st.session_state.current_result = {
                'title': title,
                'text': text_input,
                'overall_probs': overall_probs,
                'sentence_results': sentence_results,
                'timestamp': timestamp
            }
            
            # Tambah ke history
            st.session_state.history.append((
                title,
                text_input,
                overall_probs,
                timestamp
            ))
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
        finally:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

# Tampilkan hasil analisis terbaru
if st.session_state.current_result:
    result = st.session_state.current_result
    
    st.markdown("## 📊 Analysis Results")
    
    # Overall results card
    st.markdown(f"""
<div style="background-color: #111827; padding: 30px; border-radius: 12px; border: 1px solid #374151;">
    <h3 style="color: #ffffff; border-bottom: 2px solid #2563eb; padding-bottom: 10px;">
        🎯 Overall Detection Result
    </h3>
    <div style="display: flex; justify-content: space-around; margin: 30px 0;">
        <div style="
            background: #1f2937;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #4CAF50;
            color: white;
            width: 45%;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        ">
            <h2 style="color: #4CAF50;">👤 {result['overall_probs'][0]*100:.1f}%</h2>
            <p><strong>Human</strong></p>
            <p style="opacity: 0.8;">Probability this text was written by a human</p>
        </div>
        <div style="
            background: #1f2937;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #f44336;
            color: white;
            width: 45%;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        ">
            <h2 style="color: #f44336;">🤖 {result['overall_probs'][1]*100:.1f}%</h2>
            <p><strong>AI</strong></p>
            <p style="opacity: 0.8;">Probability this text was generated by AI</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    
    # Visualisasi
    st.markdown("### 📈 Visual Analysis")
    create_visualization(result['overall_probs'])
    

# History section
if st.session_state.history:
    st.markdown("## 📜 Analysis History")
    
    # Tombol clear all history
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()
    

# Tampilkan history (terbaru di atas)
for i in range(len(st.session_state.history) - 1, -1, -1):
    title, text, probs, timestamp = st.session_state.history[i]

    # Pastikan timestamp berupa objek datetime
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            pass  # fallback: tampilkan string apa adanya

    # Format timestamp dengan aman
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(timestamp, datetime) else timestamp

    with st.expander(f"📝 {title} - {timestamp_str}"):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"**🤖 AI Score:** {probs[1] * 100:.1f}%")
            st.progress(probs[1])

        with col2:
            st.markdown(f"**👤 Human Score:** {probs[0] * 100:.1f}%")
            st.progress(probs[0])

        with col3:
            if st.button("🗑️", key=f"delete_{i}", help="Delete this analysis"):
                st.session_state.history.pop(i)
                st.rerun()

        # Text preview (dark mode styled)
        st.markdown("**Text Preview:**")
        st.markdown(f"""
        <div style='
            background: #1e293b;
            color: #f1f5f9;
            padding: 15px 20px;
            border-radius: 10px;
            border-left: 5px solid #8b5cf6;
            font-size: 15px;
            line-height: 1.6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-top: 10px;
        '>
        {text[:300]}{'...' if len(text) > 300 else ''}
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧠 AI vs Human Text Detector | Built with Streamlit & Transformers"
    "</div>", 
    unsafe_allow_html=True
)
