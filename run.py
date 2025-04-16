import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer('spiece.model')

model_config = T5Config.from_json_file('config.json')

model = T5ForConditionalGeneration(model_config)

state_dict = torch.load('pytorch_model.bin', map_location=device)

model.load_state_dict(state_dict)

model.to(device)

@st.cache_resource
def text_summary(text):
    t5_prepared_Text = text
    inputs = tokenizer(t5_prepared_Text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=1024,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
    ).to(device)
    
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text


choice = st.sidebar.selectbox(
    "Select your choice", ["Summarize Text", "Summarize Document"]
)

if choice == "Summarize Text":
    st.subheader("Summarize Text using T5")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document using T5")
    input_file = st.file_uploader("Upload your document here", type=["pdf"])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)
