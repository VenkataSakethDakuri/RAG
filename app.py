import os
import streamlit as st
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv
import datetime
import io
import qdrant_client
from llama_parse import LlamaParse
import nest_asyncio
import shutil

# Apply nest_asyncio for async operations
nest_asyncio.apply()
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(page_title="University Document RAG with Qdrant", layout="wide")

# Initialize Qdrant client and collection
@st.cache_resource
def initialize_qdrant():
    try:
        # Define the path for Qdrant DB
        qdrant_path = "./qdrant_db"
        # Create a client (local)
        client = qdrant_client.QdrantClient(path=qdrant_path)
        # Create or get collection
        collection_name = "university_stats"
        # Check if collection exists, if not create it
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=3072,  # Dimensions for text-embedding-3-large
                    distance=qdrant_client.http.models.Distance.COSINE
                )
            )
        
        return client, collection_name, qdrant_path
    except Exception as e:
        st.error(f"Error initializing Qdrant: {str(e)}")
        return None, None, None

# API Key Handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
        st.stop()
    
os.environ["OPENAI_API_KEY"] = api_key

llama_api_key = os.getenv("LLAMAPARSE_API_KEY")
if not llama_api_key:
    llama_api_key = st.text_input("Enter your LlamaParse API Key:", type="password")
    if not llama_api_key:
        st.warning("Please enter your LlamaParse API Key to continue.")
        st.stop()

# Initialize Qdrant
qdrant_client, collection_name, qdrant_path = initialize_qdrant()

# Model Initialization
@st.cache_resource
def initialize_models():
    llm = OpenAI(
        system_prompt="""Extract precise numerical answers from university documents. 
        Return ONLY the number without additional text. Return NULL if unavailable after thorough search.""",
        model="gpt-4o-mini",
        temperature=0.0
    )
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=3072)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

llm, embed_model = initialize_models()

# Function to reset Qdrant DB
def reset_qdrant_db():
    try:
        if os.path.exists(qdrant_path):
            # Delete the Qdrant directory
            
            st.success("Qdrant DB reset successfully")
        
        # Reinitialize Qdrant
        new_client, new_collection_name, _ = initialize_qdrant()
        return new_client, new_collection_name
    except Exception as e:
        st.error(f"Error resetting Qdrant DB: {str(e)}")
        return None, None

# Document Processing
@st.cache_resource
def initialize_parser():
    return LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        verbose=True,
        num_workers=4
    )

parser = initialize_parser()

# File Paths
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# UI Components
st.title("University Stats RAG with Qdrant")
st.header("Upload Documents")

st.info("""
This system uses LlamaParse for enhanced document parsing and Qdrant for vector storage. Supported file types include:
- PDF documents (.pdf)
- Excel spreadsheets (.xlsx, .xls, .xlsm, .xlsb, .csv)
- Word documents (.docx, .doc, .docm, .rtf)
- PowerPoint presentations (.pptx, .ppt, .pptm)
- Text files (.txt)
- Image files (.jpg, .png, .gif)
- And many more!
""")

fixed_questions = [
    "Name of College or University",
    
    "Institution's official fall reporting date (only year)",
    
    "Number of Men who are Degree-seeking, first-time, first-year students in full time enrollment",
    "Number of Women who are Degree-seeking, first-time, first-year students in full time enrollment",
    
    "All other degree seeking undergraduate students who are Men in full time enrollment.",
    "All other degree seeking undergraduate students who are Women in full time enrollment.",
    
    "Total number of Men who are degree-seeking undergraduate students in full time enrollment.",
    "Total number of Women who are degree-seeking undergraduate students in full time enrollment.",
        
    "Degree-seeking, First-time, First-year who are Nonresidents",
    "Degree-seeking Undergraduates(include first-time, first-year) who are Nonresidents",
    "Total Undergraduates(both degree-seeking and non-degree-seeking) who are International(Nonresidents)",

    "Total number of Degree-seeking, First-time, First-year",
    "Total number of Degree-seeking Undergraduates(include first-time, first-year)",
    "Total number of Total Undergraduates(both degree-seeking and non-degree-seeking)",
    
    "Total number of men first-time, first-year students who applied",
    "Total number of women first-time, first-year students who applied",
    
    "Total number of men first-time, first-year students admitted",
    "Total number of women first-time, first-year students admitted",
        
    "Men who are Full-time, first-time, first-year students enrolled",
    "Women who are Full-time, first-time, first-year students enrolled",
    
    "Total first-time, first-year (degree seeking) who applied In-State",
    "Total first-time, first-year (degree seeking) who applied Out-of-State",
    "Total first-time, first-year (degree seeking) who applied International",
    "Total first-time, first-year (degree seeking) who applied",
    
    "Total first-time, first-year (degree seeking) who were admitted In-State",
    "Total first-time, first-year (degree seeking) who were admitted Out-of-State",
    "Total first-time, first-year (degree seeking) who were admitted International",
    "Total first-time, first-year (degree seeking) who were admitted",
    
    "Total first-time, first-year (degree seeking) enrolled In-State",
    "Total first-time, first-year (degree seeking) enrolled Out-of-State",
    "Total first-time, first-year (degree seeking) enrolled International",
    "Total first-time, first-year (degree seeking) enrolled",
    
    "Number of qualified applicants offered a place on waiting list",
    "Number accepting a place on the waiting list",
    "Number of wait-listed students admitted",
    
    "Give the relative importance of various factors for admission decisions, list all the factors along with their importance",
    
    "percent of first-time, first-year students enrolled who submitted national standardized SAT test scores.",
    "percent of first-time, first-year students enrolled who submitted national standardized ACT test scores.",
    "Number of first-time, first-year students enrolled who submitted national standardized SAT test scores",
    "Number of first-time, first-year students enrolled who submitted national standardized ACT test scores",
    
    """For each assessment listed below, report the score that represents the 25th percentile, 50th percentile and the 75th percentile score.,
    SAT Composite (400 - 1600),
    SAT Evidence-Based Reading and Writing (200 - 800),
    SAT Math (200 - 800),
    ACT Composite (0 - 36),
    ACT Math (0 - 36),
    ACT English (0 - 36),
    ACT Reading (0 - 36),
    ACT Science (0 - 36),
    ACT Writing (0 - 36)""",
    
    "Give the following in tabular format : Score Range, SAT Evidence-Based Reading and Writing, SAT Math",
    "Give the following in tabular format : Score Range, SAT Composite",
    "Give the following in tabular format : Score Range, ACT Composite",
    "Give the following in tabular format : Score Range, ACT English, ACT Math, ACT Reading, ACT Science",
    
    """Percent of all degree-seeking, first-time, first-year students who had high school class rank within each of the following ranges,
    Percent in top tenth of high school graduating class,
    Percent in top quarter of high school graduating class,
    Percent in top half of high school graduating class,
    Percent in bottom half of high school graduating class,
    Percent in bottom quarter of high school graduating class,
    Percent of total first-time, first-year students who submitted high school class rank""",
    
    """Percentage of all enrolled, degree-seeking, first-time, first-year students who had high school grade-point averages within each of the following ranges (using 4.0 scale).,
    Give these 3 columns for the score ranges listed below : Percent of students who submitted scores, Percent of students who did not submit scores, Percent of all enrolled students",
    Percent who had GPA of 4.0",
    Percent who had GPA between 3.75 and 3.99,
    Percent who had GPA between 3.50 and 3.74,
    Percent who had GPA between 3.25 and 3.49,
    Percent who had GPA between 3.00 and 3.24,
    Percent who had GPA between 2.50 and 2.99,
    Percent who had GPA between 2.0 and 2.49,
    Percent who had GPA between 1.0 and 1.99,
    Percent who had GPA below 1.0""",
    
    "Average high school GPA of all degree-seeking, first-time, first-year students who submitted GPA",
    
    "Percent of total first-time, first-year students who submitted high school GPA",
    
    "Number of early decision applications received by your institution",
    
    "Number of applicants admitted under early decision plan",
    
    "Number of early action applications received by your institution",
    
    "Number of applicants admitted under early action plan",
    
    "Number of applicants enrolled under early action plan",
    
    "give me the names of Top 5 number of degrees conferred along with their respective number of degrees conferred or percent of total degrees conferred", 
]

def process_document(doc_path, questions, client, coll_name):
    try:
        file_extractor = {ext: parser for ext in [
            ".pdf", ".txt", ".docx", ".xlsx", ".pptx", ".jpg", ".png"
        ]}
        
        file_extension = os.path.splitext(doc_path)[1].lower()
        with st.spinner(f"Parsing {os.path.basename(doc_path)}"):
            documents = SimpleDirectoryReader(
                input_files=[doc_path], 
                file_extractor=file_extractor
            ).load_data()

        # Create vector store with Qdrant
        vector_store = QdrantVectorStore(client=client, collection_name=coll_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index with Qdrant storage
        with st.spinner("Creating vector index with Qdrant..."):
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
        
        query_engine = index.as_query_engine(similarity_top_k=5)
        
        results = {}
        for question in questions:
            with st.spinner(f"Processing: {question}"):
                response = query_engine.query(question)
                results[question] = str(response)
        
        return results
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def save_results_to_excel(all_results):
    try:
        df = pd.DataFrame(index=all_results.keys())
        
        for question in fixed_questions:
            df[question] = [all_results[doc].get(question, "") for doc in all_results.keys()]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"university_stats_{timestamp}.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)
        
        df.to_excel(excel_path)
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="University Stats")
        excel_buffer.seek(0)
        
        return excel_path, excel_buffer, excel_filename
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None, None, None

# File uploader
uploaded_files = st.file_uploader(
    "Upload university documents", 
    accept_multiple_files=True, 
    type=["pdf", "txt", "docx", "xlsx", "pptx", "jpg", "png"]
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} document(s)")

if st.button("Start Processing Documents"):
    if not uploaded_files:
        st.warning("No documents uploaded. Please upload documents first.")
    else:
        with st.spinner("Saving documents to data folder..."):
            # Clear data directory
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)
            
            st.success(f"Successfully saved {len(uploaded_files)} document(s) to {data_dir}")
        
        all_results = {}
        for doc_path in saved_files:
            doc_file = os.path.basename(doc_path)
            st.subheader(f"Processing: {doc_file}")
            
            # Reset Qdrant DB between documents
            st.info("Resetting Qdrant DB for new document...")
            qdrant_client, collection_name = reset_qdrant_db()
            
            # Process document with fresh Qdrant DB
            results = process_document(doc_path, fixed_questions, qdrant_client, collection_name)
            
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
                
                st.download_button(
                    label="Download Excel Report",
                    data=excel_buffer,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
