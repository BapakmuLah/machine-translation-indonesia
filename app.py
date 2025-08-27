import streamlit as st
import pandas as pd
import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# FUNCTION TO LOAD MODEL 
@st.cache_resource
def load_model(use_gpu = True):

    # LOAD PRE-TRAINED MODEL
    english_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    english_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-id')

    indo_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
    indo_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-id-en')

    # SET GPU (IF ANY)
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    english_model.to(device)
    indo_model.to(device)

    models = {'Inggris --> Indonesia' : (english_tokenizer, english_model),
              'Indonesia --> Inggris' : (indo_tokenizer, indo_model)} 

    return models, device

# FUNCTION TO TRANSLATE 1 TEXT AT A TIME
def translate(text, model, tokenizer, device):
    encoder = tokenizer(text, return_tensors = 'pt', padding = True, truncation = True).to(device)
    with torch.no_grad():

        # TERJEMAHKAN TEXT
        generated = model.generate(**encoder)   # --> RETURN SEQUENCE ID VOCABULARY

        st.write(generated)

        # DECODE SEQUENCE VECTOR ID INTO NORMAL STRING (BASED ON VOCABULARY)
        decode = tokenizer.batch_decode(generated, skip_special_tokens = True)  # --> RETURN TRANSLATED TEXT
        

    return decode[0]


# -------------------- CODE START HERE -----------------------------#
st.set_page_config(page_title = 'Welcome Guys', page_icon = ':<')

st.title('Machine Translation')

# LOAD AND DEFINE MODEL
models, device = load_model(use_gpu = False)


tab1, tab2 = st.tabs(['Terjemahan Teks', 'Terjemahan CSV'])

with tab1:
    col1, col2 = st.columns(spec = [1.5, 3])
    
    with col1:
        choose_translate = st.selectbox(label = 'Pilih Terjemahan', options = ['Inggris --> Indonesia', 'Indonesia --> Inggris'])

    # VALIDASI LABEL TEXT AREA
    if choose_translate == 'Inggris --> Indonesia':
        text_label = 'Masukkan teks bahasa Inggris!'
    else:
        text_label = 'Masukkan teks bahasa Indonesia!'

    # TEXT AREA
    user_text = st.text_area(label = text_label, label_visibility = 'visible')

    # TEXT BUTTON
    text_btn = st.button("Terjemahkan", type = 'primary', use_container_width = False)

    # VALIDATE BUTTON 
    if text_btn:
        if not user_text.strip():
            st.write('Input tidak boleh Kosong!')
        else:

            # LOAD MODEL BASED SELECT BOX
            tokenizer, model = models[choose_translate]

            # CREATE LOADING SPINNER
            with st.spinner(text = 'Please Wait!'):

                t0 = time.time()    # ---> CURRENT TIME

                translate_result = translate(text = user_text, 
                                             model = model, 
                                             tokenizer = tokenizer, 
                                             device = device)
                
                # CALCULATE TIME TAKEN TO TRANSLATE A TEXT
                dt = time.time() - t0

            st.success(f"✅ Terjemahan (selesai {dt:.2f} detik)")
            st.markdown(f"**{translate_result}**")



with tab2:
    csv_file = st.file_uploader(label = "Upload CSV dengan kolom bernama 'text' atau 'teks'", type = ['CSV'])

    if csv_file is not None:
        df = pd.read_csv(csv_file)

        # VALIDASI APAKAH TERDAPAT KOLOM 'text / teks'
        col_name = 'text' if 'text' in df.columns else ('teks' if 'teks' in df.columns else None)

        if col_name is None:
            st.error(body = "CSV harus memiliki kolom bernama `text` atau `teks`.")
        else:
            st.write('📊 Preview data')
            st.dataframe(df.head())

            run = st.button(label = 'Terjemahkan CSV',  type = 'primary')

            if run:
                st.write('congrats')

