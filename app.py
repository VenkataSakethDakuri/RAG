import os
import streamlit as st
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import datetime
import io

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Document RAG System", layout="wide")

# Set up OpenAI credentials
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
        st.stop()
    
os.environ["OPENAI_API_KEY"] = api_key

# Initialize LLM and embedding model
@st.cache_resource
def initialize_models():
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

llm, embed_model = initialize_models()

# Define data directory
data_dir = "C:\\Users\\DELL\\OneDrive\\Desktop\\VS CODE\\RAG\\data"

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Create output directory for Excel files
output_dir = os.path.join(os.path.dirname(data_dir), "output")
os.makedirs(output_dir, exist_ok=True)

# Document upload section
st.title("Document RAG System")
st.header("Upload Documents")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt", "docx"])

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} document(s)")

# Define fixed questions
fixed_questions = [
    "What is the main topic of this document?",
    "What are the key points discussed?",
    "Are there any action items mentioned?",
]

# Function to process a single document
def process_document(doc_path, questions):
    try:
        # Load single document
        documents = SimpleDirectoryReader(input_files=[doc_path]).load_data()
        st.write(f"Processing: {os.path.basename(doc_path)}")
        
        # Create index for this document
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
        
        # Process fixed questions
        results = {}
        for question in questions:
            with st.spinner(f"Answering: {question}"):
                response = query_engine.query(question)
                results[question] = str(response)
        
        return results
    except Exception as e:
        st.error(f"Error processing document {os.path.basename(doc_path)}: {str(e)}")
        return None

# Function to save results to Excel
def save_results_to_excel(all_results):
    try:
        # Create dataframe with document names as index
        df = pd.DataFrame(index=all_results.keys())
        
        # Add columns for each question
        for question in fixed_questions:
            # Extract answers for this question from all documents
            df[question] = [all_results[doc].get(question, "") for doc in all_results.keys()]
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"document_answers_{timestamp}.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)
        
        # Save to Excel file
        df.to_excel(excel_path)
        
        # Also create a downloadable version for the user
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Document Answers")
        excel_buffer.seek(0)
        
        return excel_path, excel_buffer, excel_filename
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None, None, None

# Start processing button
if st.button("Start Processing Documents"):
    if not uploaded_files:
        st.warning("No documents uploaded. Please upload documents first.")
    else:
        # Save uploaded files to data directory automatically
        with st.spinner("Saving documents to data folder..."):
            # Clear existing files in data directory
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Save new files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)
            
            st.success(f"Successfully saved {len(uploaded_files)} document(s) to {data_dir}")
        
        # Process each document individually
        all_results = {}
        for doc_path in saved_files:
            doc_file = os.path.basename(doc_path)
            st.subheader(f"Processing: {doc_file}")
            
            # Process the document
            results = process_document(doc_path, fixed_questions)
            
            if results:
                # Display results for this document
                for question, answer in results.items():
                    st.write(f"**Q: {question}**")
                    st.write(f"A: {answer}")
                    st.write("---")
                
                all_results[doc_file] = results
        
        if all_results:
            st.success("All documents processed successfully!")
            
            # Save results to Excel
            excel_path, excel_buffer, excel_filename = save_results_to_excel(all_results)
            
            if excel_path:
                st.success(f"Results saved to Excel file: {excel_path}")
                
                # Add download button
                st.download_button(
                    label="Download Excel Report",
                    data=excel_buffer,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
