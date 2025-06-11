# Chat-with-PDF-using-Groq-LLM
📝 Description
This application enables natural language interaction with the contents of uploaded PDF documents. Once the PDFs are uploaded, the text is extracted, chunked, and embedded using sentence-transformers. The system then stores these embeddings in a Chroma vector database and retrieves the most relevant chunks to answer user queries using LangChain’s QA chain and Groq’s LLM.

It supports:

Multi-page and multi-document upload

Free and local embedding generation (no Hugging Face token needed)

Accurate, grounded answers with reasoning

Clean and simple UI using Streamlit

🚀 Features
📚 Upload multiple PDF documents

🔍 Ask questions and get answers from the content

🤖 Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings

🧠 Powered by Groq's Gemma2-9b-It model (switchable to LLaMA3)

💾 Embeddings stored in ChromaDB for semantic search

💬 Streamlit UI for easy user interaction

🔐 .env file support for secure API key management

📁 Project Structure:
```.
├── app.py                 # Main Streamlit application
├── chromadb_index/        # Stores Chroma vector index
├── .env                   # Contains API keys
├── README.md              # Project documentation
 ```
🔑 Environment Setup
Create a .env file in your root folder and add
```GROQ_API_KEY=your_groq_api_key_here```

Install dependencies:
```pip install -r requirements.txt```

▶️ Run the App
```streamlit run app.py```
💡 How It Works
PDF Upload: Upload one or more PDF files via the sidebar.

Text Extraction: Extracts all text from uploaded PDFs.

Chunking: Splits text into manageable chunks using RecursiveCharacterTextSplitter.

Embedding: Generates vector embeddings locally using MiniLM.

Storage: Embeddings are stored in ChromaDB (chromadb_index/).

Querying: User enters a question, relevant chunks are retrieved.

LLM Response: Groq LLM answers based on retrieved content using a custom prompt.



