import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_uploaded_file(uploaded_file):
    """Process uploaded file and return documents"""
    if uploaded_file is None:
        st.error("Please upload a file first")
        st.stop()

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp:
        temp.write(uploaded_file.read())
        file_path = temp.name

    try:
        # Determine file type and load accordingly
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext == "pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_ext == "txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_ext == "docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.error("Unsupported file type. Please upload PDF, TXT, or DOCX files.")
            st.stop()

        docs = loader.load()
        
        # Clean up temporary file
        os.unlink(file_path)
        
        if not docs:
            st.error("No content found in the uploaded file")
            st.stop()
            
        return docs
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(file_path):
            os.unlink(file_path)
        st.error(f"Error processing file: {str(e)}")
        st.stop()

def docs_chunks(docs):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        st.error("No chunks created from the document")
        st.stop()
        
    return chunks

def vector_db(chunks):
    """Create vector database from chunks"""
    try:
        embeddings = OpenAIEmbeddings()
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        db = FAISS.from_documents(chunks, embeddings)
        return db
        
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        st.stop()

def get_llm_response(db, style_input, length_input, query="Summarize this research paper"):
    """Generate response using RAG chain"""
    try:    
        # Create retriever and get relevant documents
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        )
        
        # Retrieve relevant documents
        relevant_docs = retriever.invoke(query)
        
        # Format documents into context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Initialize the model
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )

        # Create the prompt template
        template = """You are a research paper summarization expert. Based on the provided context from a research paper, create a comprehensive summary with the following specifications:

        Context from the research paper: {context}
        Explanation Style: {style_input}
        Explanation Length: {length_input}

        Instructions:
        1. Mathematical Details: Include relevant mathematical equations if present in the paper and explain mathematical concepts clearly.
        2. Use relatable analogies to simplify complex ideas when appropriate.
        3. Structure your response clearly with proper sections.
        4. If certain information is not available in the provided context, respond with "Insufficient information available" for that specific aspect.
        """

        # template="""
        # Please summarize the research paper from given context {context} with the following specifications:
        # Explanation Style: {style_input}  
        # Explanation Length: {length_input}  
        # 1. Mathematical Details:  
        # - Include relevant mathematical equations if present in the paper.  
        # - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
        # 2. Analogies:  
        # - Use relatable analogies to simplify complex ideas.  
        # If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
        # Ensure the summary is clear, accurate, and aligned with the provided style and length.
        # """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "style_input", "length_input"]
        )

        # Create a simple chain
        chain = prompt | model | StrOutputParser()

        # Generate response with properly formatted inputs
        result = chain.invoke({
            "context": context,
            "style_input": style_input,
            "length_input": length_input
        })
        
        return result
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Research Paper Summarizer", page_icon="📚", layout="centered") #wide
    
    st.title("📚 Research Paper Summarizer")
    st.markdown("Upload a research paper and get a customized summary based on your preferences.")
    
    # Sidebar for configuration
    # st.sidebar.header("Configuration")
    st.header("Configuration")

    
    # File upload
    # uploaded_file = st.sidebar.file_uploader(
    #     "Upload your research paper", 
    #     type=["pdf", "txt", "docx"]
    # )
    
    uploaded_file = st.file_uploader(
        "Upload your research paper", 
        type=["pdf", "txt", "docx"]
    )

    # Style and length selection
    style_input = st.selectbox(
        "Select Explanation Style", 
        ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
        help="Choose how you want the summary to be explained"
    )
    
    length_input = st.selectbox(
        "Select Explanation Length", 
        ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
        help="Choose the length of the summary"
    )

    # Summarize button
    if st.button("🔍 Generate Summary", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a file first!")
        else:
            with st.spinner("Processing document and generating summary..."):
                try:
                    # Process the uploaded file
                    docs = get_uploaded_file(uploaded_file)
                    
                    chunks = docs_chunks(docs)
                    
                    db = vector_db(chunks)
                    
                    result = get_llm_response(db, style_input, length_input)
                    
                    if result:
                        st.success("Summary generated successfully!")
                        
                        # Display the result
                        st.subheader("📋 Summary")
                        st.markdown(result)
                        
                        # Add download button for the summary
                        st.download_button(
                            label="📥 Download Summary",
                            data=result,
                            file_name=f"summary_{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()