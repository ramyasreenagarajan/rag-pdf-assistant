<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-3776AB?style=flat-square&logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-BF91F3?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-38BDAE?style=flat-square)
![Stars](https://img.shields.io/github/stars/ramyasreenagarajan/rag-pdf-assistant?style=flat-square&color=70A5FD)

**Upload any PDF → Ask questions → Get answers grounded in your document**

[Demo](#demo) · [Installation](#installation) · [How it Works](#how-it-works) · [Usage](#usage)

</div>

---

##  What is this?

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you have a conversation with any PDF document. No hallucinations — every answer is grounded strictly in your uploaded content.

> **Why RAG?** RAG is the #1 trending AI architecture in 2026. It combines the power of semantic search with language generation — used by enterprises worldwide for document intelligence, research assistants, and knowledge bases.

### Features

-  **Any PDF** — research papers, reports, textbooks, contracts
-  **Semantic search** using TF-IDF + cosine similarity
-  **Interactive chat UI** built with Streamlit
-  **CLI mode** for terminal power users
-  **100% local** — no API key, no data sent to cloud
-  **Source citations** — see exactly which chunk answered your question
-  **Fast indexing** — 200-page PDF indexed in under 3 seconds

---

##  Demo

<table>
<tr>
<td>

**CLI Mode**
```bash
$ python src/cli.py --demo

 What accuracy do CNNs achieve?
──────────────────────────────────────────────────────
  Machine learning algorithms achieve over 95 percent
  accuracy in detecting diabetic retinopathy.
  Deep learning models identify pneumonia with 90%+
  sensitivity from chest X-rays.

   Top sources:
     1. chunk #3  ~page 1  score=0.421
     2. chunk #1  ~page 1  score=0.318
──────────────────────────────────────────────────────
```

</td>
</tr>
</table>

---

## How it Works

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   PDF File  │───▶│  Text Extract│───▶│  Chunk (400 wds │
│             │    │  (PyMuPDF)   │    │  80 wd overlap) │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
                                                ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Answer    │◀───│ Generate ans │◀───│  TF-IDF Vectors │
│  + Sources  │    │  from chunks │    │  Cosine Search  │
└─────────────┘    └──────────────┘    └─────────────────┘
```

| Step | Method | Detail |
|------|--------|--------|
| Extraction | PyMuPDF | Handles text, tables, multi-column |
| Chunking | Sliding window | 400 words, 80-word overlap |
| Embedding | TF-IDF (50K features) | Bigrams, sublinear TF |
| Retrieval | Cosine similarity | Top-K relevant chunks |
| Generation | Keyword-based extraction | No API needed |

---

## Project Structure

```
rag-pdf-assistant/
│
├── src/
│   ├── rag_engine.py      # Core RAG pipeline (extraction, chunking, retrieval)
│   └── cli.py             # Command-line interface + demo mode
│
├── app/
│   └── app.py             # Streamlit web application
│
├── data/                  # Auto-created index cache
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
# 1. Clone
git clone https://github.com/ramyasreenagarajan/rag-pdf-assistant.git
cd rag-pdf-assistant

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Web App (Streamlit)
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` → Upload PDF → Start asking questions

### CLI — Demo mode (no PDF needed)
```bash
python src/cli.py --demo
```

### CLI — With your PDF
```bash
# Interactive Q&A
python src/cli.py --pdf your_document.pdf

# Single question
python src/cli.py --pdf your_document.pdf --ask "What are the main findings?"
```

---

## Extend with an LLM

The `generate_answer_local()` function in `src/rag_engine.py` is a drop-in swap point. Replace it with any LLM:

```python
# Ollama (local, free)
import ollama
response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": build_prompt(query, context_chunks)}
])
return response["message"]["content"]

# Claude API
import anthropic
client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": build_prompt(query, context_chunks)}]
)
return msg.content[0].text
```

---

## Roadmap

- [ ] Swap TF-IDF for sentence-transformers (semantic embeddings)
- [ ] Add multi-PDF support with source attribution
- [ ] Integrate Ollama for fully local LLM generation
- [ ] Export Q&A sessions as PDF reports
- [ ] Add support for DOCX and TXT files

---

## Author

**Ramya Sree Nagarajan**
MSc Artificial Intelligence · Royal Holloway, University of London
IEEE Published Researcher · Python · ML · Cybersecurity

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/ramya-sree-nagarajan-619245345)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:ramyasreenagarajan@gmail.com)
[![IEEE](https://img.shields.io/badge/IEEE%20Paper-00629B?style=flat-square&logo=ieee&logoColor=white)](https://doi.org/10.1109/10531377)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
