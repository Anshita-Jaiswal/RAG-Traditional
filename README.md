# RAG-Traditional

**Retrieval-Augmented Generation (RAG) built the "traditional" way — using [Typesense](https://typesense.org/) as the vector store, with a bonus LangChain-powered pipeline for full end-to-end RAG.**

This project is a hands-on, notebook-driven walkthrough of building RAG systems from first principles. It shows two approaches side by side:

1. **Native Typesense client** — manually defining a schema, ingesting structured data (a book catalog), and running keyword/faceted/vector-style search directly against Typesense.
2. **LangChain + Typesense + Groq** — a full RAG pipeline that loads a raw text document, splits it into chunks, embeds it with HuggingFace sentence-transformers, stores it in Typesense as a LangChain vector store, retrieves relevant chunks for a query, and (optionally) feeds that context to a Groq-hosted LLM.

---

## Table of Contents

- [Why "Traditional"?](#why-traditional)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the repo](#1-clone-the-repo)
  - [2. Install dependencies](#2-install-dependencies)
  - [3. Set up Typesense](#3-set-up-typesense)
  - [4. Configure environment variables](#4-configure-environment-variables)
- [Usage](#usage)
  - [Part 1 — Native Typesense: books search](#part-1--native-typesense-books-search)
  - [Part 2 — LangChain RAG pipeline](#part-2--langchain-rag-pipeline)
- [Example Queries](#example-queries)
- [Contributing](#contributing)
- [License](#license)

---

## Why "Traditional"?

Most RAG tutorials reach straight for a purpose-built vector database (Pinecone, Weaviate, Qdrant). This project instead uses **Typesense**, a fast, typo-tolerant, open-source search engine that also supports vector/semantic search — showing that you don't need a dedicated vector DB to build a working RAG system. It's "traditional" in the sense of leaning on a search-engine-first mental model (schemas, fields, facets, filters) rather than a pure embeddings-only approach.

## Architecture

```
                     ┌─────────────────────────┐
                     │      books.jsonl         │
                     │  (structured book data)  │
                     └────────────┬─────────────┘
                                  │  import
                                  ▼
┌────────────┐      schema      ┌─────────────────────────┐
│  Typesense  │◄────────────────│   Typesense Client (raw) │
│   Cloud /   │                 └─────────────────────────┘
│   Local     │                            │
│  Collection │◄─── search/filter/facet ───┘
└─────┬───────┘
      │
      │  (also used as a LangChain VectorStore)
      ▼
┌─────────────────┐   split    ┌───────────────────┐   embed   ┌──────────────────────┐
│   Google.txt      ├──────────►│ CharacterTextSplitter ├─────────►│ HuggingFaceEmbeddings │
└─────────────────┘            └───────────────────┘           └──────────┬────────────┘
                                                                            │
                                                                            ▼
                                                              ┌─────────────────────────┐
                                                              │  Typesense (LangChain)   │
                                                              │      Vector Store        │
                                                              └────────────┬────────────┘
                                                                           │ similarity_search / retriever
                                                                           ▼
                                                              ┌─────────────────────────┐
                                                              │   Retrieved Context      │
                                                              └────────────┬────────────┘
                                                                           │
                                                                           ▼
                                                              ┌─────────────────────────┐
                                                              │   Groq LLM (ChatGroq)    │
                                                              │   → Final RAG answer     │
                                                              └─────────────────────────┘
```

## Repository Structure

```
RAG-Traditional/
├── data/                # Supplementary/raw data assets used across the notebooks
├── notebook/             # Additional exploratory notebooks
├── typesense.ipynb       # Main notebook: end-to-end walkthrough (start here)
├── main.py                # Minimal entry-point stub
├── books.jsonl            # Sample structured dataset (books catalog) for Typesense ingestion
├── Google.txt              # Sample unstructured text document used for the LangChain RAG demo
├── requirements.txt         # Pip-based dependency list
├── pyproject.toml            # Project metadata & dependencies (uv-managed)
├── uv.lock                    # Locked dependency versions (uv)
├── .python-version              # Pinned Python version (3.13)
└── .env                          # Environment variables (API keys — see Security Note)
```

## Tech Stack

| Layer | Tool |
|---|---|
| Vector / search store | [Typesense](https://typesense.org/) (Cloud or self-hosted) |
| Orchestration | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-community`, `langchain-core`, `langchain-text-splitters`) |
| Embeddings | `langchain-huggingface` / `sentence-transformers` |
| LLM | [Groq](https://groq.com/) via `langchain-groq` (`ChatGroq`) |
| Document loading | `TextLoader`, `pypdf`, `pymupdf` |
| Alternate local vector stores (available, optional) | `faiss-cpu`, `chromadb` |
| Environment / config | `python-dotenv` |
| Package management | [`uv`](https://docs.astral.sh/uv/) (preferred) or `pip` |
| Runtime | Python 3.13 |

## Prerequisites

- Python **3.13+**
- A [Typesense Cloud](https://cloud.typesense.org/) cluster (free tier works) **or** a local Typesense server
- A [Groq API key](https://console.groq.com/keys) (free tier available) for the LLM step
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed (optional but recommended — the repo ships a `uv.lock`)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Anshita-Jaiswal/RAG-Traditional.git
cd RAG-Traditional
```

### 2. Install dependencies

**Using `uv` (recommended, matches `uv.lock`):**

```bash
uv sync
```

**Using `pip`:**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` includes a small typo (`python-dotenvuv`) — if `pip install -r requirements.txt` fails on that line, install `python-dotenv` and `uv` separately, or just use `pyproject.toml` via `uv sync` instead.

### 3. Set up Typesense

**Option A — Typesense Cloud (used in the notebook)**

1. Create a free cluster at [cloud.typesense.org](https://cloud.typesense.org/).
2. Once initialized, create a collection (the notebook does this in code, or you can do it from the dashboard).
3. Note your cluster's host (e.g. `xxxx.a1.typesense.net`), port (`443`), protocol (`https`), and Admin API key.

**Option B — Local Typesense (via Docker)**

```bash
docker run -p 8108:8108 -v/tmp/typesense-data:/data typesense/typesense:latest \
  --data-dir /data --api-key=xyz --enable-cors
```

Then use `'host': 'localhost'`, `'port': '8108'`, `'protocol': 'http'` when creating the client.

### 4. Configure environment variables

Create a `.env` file in the project root (**do not commit this file** — see the [Security Note](#️-security-note) below):

```env
TYPESENSE_HOST=your-cluster.a1.typesense.net
TYPESENSE_PORT=443
TYPESENSE_PROTOCOL=https
TYPESENSE_API_KEY=your_typesense_admin_api_key
GROQ_API_KEY=your_groq_api_key
```

Then load them in the notebook instead of hardcoding values:

```python
import os
from dotenv import load_dotenv

load_dotenv()

client = typesense.Client({
    'nodes': [{
        'host': os.environ['TYPESENSE_HOST'],
        'port': os.environ['TYPESENSE_PORT'],
        'protocol': os.environ['TYPESENSE_PROTOCOL'],
    }],
    'api_key': os.environ['TYPESENSE_API_KEY'],
    'connection_timeout_seconds': 2
})
```

## Usage

Launch Jupyter and open the main notebook:

```bash
uv run jupyter notebook typesense.ipynb
# or, with pip/venv:
jupyter notebook typesense.ipynb
```

### Part 1 — Native Typesense: books search

Walks through:

- Creating a `typesense.Client` connection
- Defining a `books` collection schema (`title`, `authors`, `publication_year`, `ratings_count`, `average_rating`)
- Bulk-importing records from `books.jsonl`
- Running searches with `query_by`, `sort_by`, `filter_by`, and `facet_by` (e.g. searching "harry potter", filtering by `publication_year`, faceting by `authors`)

This section demonstrates that Typesense's search primitives (full-text, filter, facet, sort) can act as the "retrieval" half of RAG even without embeddings.

### Part 2 — LangChain RAG pipeline

Walks through:

1. Loading `Google.txt` with `TextLoader`
2. Splitting it into chunks with `CharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=100`)
3. Generating embeddings with `HuggingFaceEmbeddings`
4. Indexing the chunks into Typesense as a LangChain `VectorStore` (`Typesense.from_documents`)
5. Running `similarity_search()` and using `.as_retriever()` to fetch relevant chunks for a query
6. (Extend it yourself) Piping retrieved context into `ChatGroq` to generate a grounded LLM answer

## Example Queries

```python
# Faceted/filtered search over structured book data
search_parameters = {
    'q': 'harry potter',
    'query_by': 'title',
    'filter_by': 'publication_year:<1998',
    'sort_by': 'publication_year:desc'
}
client.collections['books'].documents.search(search_parameters)
```

```python
# Semantic similarity search over unstructured text (LangChain)
query = "What is artificial intelligence"
found_docs = docsearch.similarity_search(query)
print(found_docs[0].page_content)
```


## Contributing

Issues and pull requests are welcome. If you're extending this project, please avoid committing real API keys or secrets — use `.env` and `python-dotenv` as described above.

## License

No license file is currently included in this repository. Until one is added, all rights are reserved by the repository owner — check with [@Anshita-Jaiswal](https://github.com/Anshita-Jaiswal) before reusing this code.
