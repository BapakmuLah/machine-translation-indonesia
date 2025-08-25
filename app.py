import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import io, time, math

# ================================
# Config & Sidebar Options
# ================================
st.set_page_config(page_title="English → Indonesian Translator", page_icon="🌍", layout="wide")
st.title("🌍 English → Indonesian Translator")
st.write("Terjemahkan teks Inggris → Indonesia. Dua mode: input teks langsung & upload CSV.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    max_length = st.number_input("Max length", min_value=32, max_value=512, value=256, step=8)
    num_beams = st.number_input("Num beams (beam search)", min_value=1, max_value=8, value=4, step=1)
    batch_size = st.number_input("Batch size (CSV)", min_value=1, max_value=128, value=16, step=1)
    prefer_gpu = st.checkbox("Gunakan GPU bila tersedia", value=True)
    show_time_each_batch = st.checkbox("Tampilkan waktu tiap batch", value=True)

# ================================
# Load Model (cached)
# ================================
@st.cache_resource
def load_model_and_device(use_gpu=True):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model_and_device(prefer_gpu)

# ================================
# Utilities
# ================================
def translate_batch(texts, max_length, num_beams, device):
    """Translate a list[str] in one forward pass."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_length=int(max_length),
            num_beams=int(num_beams),
        )
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return out

def translate_in_batches(texts, batch_size, max_length, num_beams, device, progress_container=None):
    """
    Streamlit-friendly 'tqdm': update progress & ETA per batch.
    Returns list of translations in the original order.
    """
    n = len(texts)
    if n == 0:
        return []

    # UI elements
    if progress_container is None:
        progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status = progress_container.empty()

    results = [None] * n
    total_batches = math.ceil(n / batch_size)
    t0 = time.time()
    last_time = t0

    for b in range(total_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)
        batch_texts = [str(x) for x in texts[start:end]]

        # translate this batch
        t_batch0 = time.time()
        outs = translate_batch(batch_texts, max_length=max_length, num_beams=num_beams, device=device)
        t_batch1 = time.time()

        # fill results
        results[start:end] = outs

        # progress & ETA
        done = end
        frac = done / n
        elapsed = t_batch1 - t0
        # Average sec per item
        sec_per_item = elapsed / max(done, 1)
        remaining = (n - done) * sec_per_item
        mm, ss = divmod(int(remaining), 60)
        eta_txt = f"ETA ~ {mm:02d}:{ss:02d}"

        # Optional per-batch timing
        batch_time = (t_batch1 - t_batch0)
        batch_info = f" | batch {b+1}/{total_batches} ({end-start} items) {batch_time:.2f}s" if show_time_each_batch else ""
        status.markdown(f"**Menerjemahkan…** {done}/{n} selesai ({frac*100:.1f}%) • {eta_txt}{batch_info}")
        progress_bar.progress(min(int(frac * 100), 100))
        last_time = t_batch1

    status.markdown(f"✅ **Selesai** • {n}/{n} baris diterjemahkan • Total {time.time() - t0:.2f}s")
    progress_bar.progress(100)
    return results

# ================================
# UI Tabs
# ================================
tab1, tab2 = st.tabs(["✍️ Terjemahan Teks", "📂 Terjemahan CSV"])

# -------- TAB 1: Single text ----------
with tab1:
    st.subheader("✍️ Input Teks")
    user_text = st.text_area("Masukkan teks Bahasa Inggris:", height=150,
                             placeholder="Contoh: The ancient manuscript contained esoteric symbols ...")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        go = st.button("Terjemahkan", type="primary", use_container_width=True, key="translate_text_btn")
    with col_b:
        _ = st.caption(f"Model: Helsinki-NLP/opus-mt-en-id • Device: **{device.upper()}**")

    if go:
        if not user_text.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menerjemahkan…"):
                out = translate_batch([user_text], max_length=max_length, num_beams=num_beams, device=device)[0]
            st.success("✅ Terjemahan")
            st.markdown(f"**{out}**")

# -------- TAB 2: CSV ----------
with tab2:
    st.subheader("📂 Upload File CSV")
    uploaded_file = st.file_uploader("Upload CSV dengan kolom `text` atau `teks`:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # kolom teks
        col_name = "text" if "text" in df.columns else ("teks" if "teks" in df.columns else None)
        if col_name is None:
            st.error("CSV harus memiliki kolom bernama `text` atau `teks`.")
        else:
            st.write("📊 Preview Data:")
            st.dataframe(df.head())

            run = st.button("🚀 Terjemahkan CSV", type="primary")
            if run:
                progress_box = st.container()
                with st.spinner("Menyiapkan batching & model…"):
                    texts = df[col_name].astype(str).tolist()

                # terjemahan dengan progress mirip tqdm
                translations = translate_in_batches(
                    texts,
                    batch_size=int(batch_size),
                    max_length=int(max_length),
                    num_beams=int(num_beams),
                    device=device,
                    progress_container=progress_box,
                )

                df_out = df.copy()
                df_out["translation"] = translations
                st.success("✅ Selesai! Kolom `translation` ditambahkan.")
                st.dataframe(df_out.head())

                # Download
                csv_buf = io.BytesIO()
                df_out.to_csv(csv_buf, index=False)
                csv_buf.seek(0)
                st.download_button(
                    "💾 Download Hasil Terjemahan",
                    data=csv_buf,
                    file_name="translation_results.csv",
                    mime="text/csv",
                )

