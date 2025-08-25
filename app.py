import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import io

# ================================
# Load Model
# ================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    return tokenizer, model

tokenizer, model = load_model()

# Function untuk terjemahan
def translate_text(texts, max_length=256):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ================================
# UI Streamlit
# ================================
st.set_page_config(page_title="English → Indonesian Translator", page_icon="🌍", layout="wide")

st.title("🌍 English → Indonesian Translator")
st.write("Gunakan aplikasi ini untuk menerjemahkan teks dari **Bahasa Inggris ke Bahasa Indonesia**. "
         "Tersedia dua fitur: input teks langsung atau upload file CSV.")

# Tabs untuk memisahkan fitur
tab1, tab2 = st.tabs(["✍️ Terjemahan Teks", "📂 Terjemahan CSV"])

# ---------------- TAB 1 -----------------
with tab1:
    st.subheader("✍️ Input Teks")
    user_text = st.text_area("Masukkan teks Bahasa Inggris:", height=150, placeholder="Contoh: The weather is beautiful today.")
    
    if st.button("Terjemahkan", key="translate_text"):
        if user_text.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Menerjemahkan..."):
                translation = translate_text([user_text])[0]
            st.success("✅ Terjemahan:")
            st.write(f"**{translation}**")

# ---------------- TAB 2 -----------------
with tab2:
    st.subheader("📂 Upload File CSV")
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom `text` atau `teks`:", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Pastikan ada kolom teks
        col_name = None
        if "text" in df.columns:
            col_name = "text"
        elif "teks" in df.columns:
            col_name = "teks"
        else:
            st.error("CSV harus memiliki kolom bernama `text` atau `teks`!")
        
        if col_name:
            st.write("📊 Preview Data:")
            st.dataframe(df.head())
            
            if st.button("Terjemahkan CSV", key="translate_csv"):
                with st.spinner("Menerjemahkan seluruh teks..."):
                    df["translation"] = translate_text(df[col_name].astype(str).tolist())
                
                st.success("✅ Selesai! Hasil terjemahan ditambahkan ke kolom `translation`.")
                st.dataframe(df.head())
                
                # Download file hasil
                buffer = io.BytesIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Hasil Terjemahan",
                    data=buffer,
                    file_name="translation_results.csv",
                    mime="text/csv"
                )
