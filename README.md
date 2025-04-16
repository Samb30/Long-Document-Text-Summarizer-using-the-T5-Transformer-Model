# Long Document Text Summarizer using the T5 Transformer Model

This project implements an **abstractive text summarization** system for long documents using the **T5 (Text-To-Text Transfer Transformer)** model. It is designed to handle lengthy documents such as research papers and articles, providing comprehensive and coherent summaries while preserving key information.

---

## ✨ Features

- ✅ Efficient processing of long documents (e.g., research papers, articles)
- ✅ Abstractive summarization capability
- ✅ Fine-tuned T5 transformer model (`LD-T5`)
- ✅ Higher ROUGE scores compared to baseline models
- ✅ Support for various document formats, including PDFs
- ✅ User-friendly interface built with **Streamlit**

---

## 🧠 Model Architecture

The project utilizes the **T5-base** model with an **encoder-decoder** structure:

- **Encoder:** Processes the input document and creates contextual representations
- **Decoder:** Generates the summary based on the encoded representations
- **Multi-head Self-Attention:** Enables the model to consider multiple perspectives and capture long-range dependencies
