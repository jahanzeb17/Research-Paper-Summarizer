# 📚 Research Paper Summarizer

This project is a Streamlit-based web application that allows you to upload research papers (PDF, TXT, DOCX) and generate customized summaries. Leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) principles, it provides flexible summarization options based on explanation style and length.

---

## ✨ Features

* **Multi-Format Support**: Upload research papers in PDF, TXT, or DOCX formats.
* **Customizable Explanation Style**: Choose from "Beginner-Friendly", "Technical", "Code-Oriented", or "Mathematical" explanation styles for your summary.
* **Adjustable Summary Length**: Select desired summary length: "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", or "Long (detailed explanation)".
* **Intelligent Summarization**: The application extracts relevant information, including mathematical equations, and uses analogies to simplify complex concepts.
* **Structured Output**: Summaries are well-structured for easy readability.
* **Error Handling**: Robust error handling for file uploads and processing.
* **Download Summary**: Easily download the generated summary as a text file.

---

## 🛠️ Technologies Used

The project is built using the following key technologies:

* **Streamlit**: For building the interactive web application interface.
* **LangChain**: A framework for developing applications powered by language models, used for document loading, text splitting, embeddings, and RAG chain construction.
* **PyMuPDFLoader**: For loading PDF documents.
* **TextLoader**: For loading plain text documents.
* **UnstructuredWordDocumentLoader**: For loading DOCX documents.
* **RecursiveCharacterTextSplitter**: For breaking down documents into manageable chunks.
* **OpenAI Embeddings**: For converting text chunks into vector representations.
* **FAISS**: A library for efficient similarity search and clustering of dense vectors, used as the vector store.
* **ChatGroq**: For integrating with Groq's low-latency inference engine, utilizing the `llama-3.3-70b-versatile` model for summarization.
* **python-dotenv**: For managing environment variables.

---

## 📺 Demo / Screenshots

**Application Interface:**
![Research Paper Summarizer Interface](Images/post.png "Main application interface")

**Summary Generation:**
![Generated Summary Example](Images/post2.png "Example of a generated summary")

---

## 🚀 Setup and Installation

Follow these steps to set up and run the Research Paper Summarizer locally:

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone <repository_url> # Replace <repository_url> with your actual repository URL
cd research-paper-summarizer # Navigate into the project directory
