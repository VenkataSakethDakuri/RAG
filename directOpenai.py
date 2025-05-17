import streamlit as st
import os
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import io

# Set page configuration
st.set_page_config(page_title="Multi-PDF Data Extraction", layout="centered")
st.title("Extract Data from Multiple PDFs using GPT-4o")

# Create .env file if it doesn't exist
if not os.path.exists('.env'):
    with open('.env', 'w') as f:
        f.write("OPENAI_API_KEY=your_openai_api_key_here")
    st.warning("Created .env file. Please edit it with your actual OpenAI API key.")

# Load environment variables from .env file
load_dotenv()

# PDF Helper function
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# OpenAI Helper function
def get_openai_response(user_query, pdf_text):
    try:
        # Initialize OpenAI client with the API key
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Create the chat completion request with modified system prompt
        messages = [
            {"role": "system", "content": "You are a data extraction assistant. Provide only numerical values from the PDF content. Do not include explanations. If the answer is not available, respond with NULL."},
            {"role": "user", "content": f"PDF Content: {pdf_text}\n\nExtract this information: {user_query}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Check if API key exists
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    st.error("OpenAI API key not found or not set. Please add it to your .env file.")
    st.stop()

# Initialize session state for results
if "all_results" not in st.session_state:
    st.session_state.all_results = {}

fixed_questions = [
    "Give me the Name of College or University",
    
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
    
  "What is the 25th, 50th, 75th percentile SAT Composite(400-1600) score ?",
  "What is the 25th, 50th, 75th percentile ACT Composite(0-36) score ?"
    
    "Give the following in tabular format : Score Range, SAT Evidence-Based Reading and Writing, SAT Math",
    "Give the following in tabular format : Score Range, SAT Composite",
    "Give the following in tabular format : Score Range, ACT Composite",
    "Give the following in tabular format : Score Range, ACT English, ACT Math, ACT Reading, ACT Science",
    
    
    "Number of early decision applications received by your institution",
    
    "Number of applicants admitted under early decision plan",
    
    "Number of early action applications received by your institution",
    
    "Number of applicants admitted under early action plan",
    
    "Number of applicants enrolled under early action plan",
    
    "give me the names of Top 5 number of degrees conferred along with their respective number of degrees conferred or percent of total degrees conferred", 
]

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

# Process PDFs and export to Excel
if uploaded_files and st.button("Extract Data from All PDFs"):
    all_results = {}
    
    # Create a progress bar for overall progress
    overall_progress = st.progress(0)
    file_status = st.empty()
    
    for file_idx, uploaded_file in enumerate(uploaded_files):
        file_status.info(f"Processing file {file_idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        # Extract text from PDF
        with st.spinner(f"Extracting text from {uploaded_file.name}..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            
        # Process each question for this PDF
        file_results = {}
        question_progress = st.progress(0)
        
        for i, question in enumerate(fixed_questions):
            with st.spinner(f"Processing question {i+1}/{len(fixed_questions)} for {uploaded_file.name}..."):
                response = get_openai_response(question, pdf_text)
                file_results[question] = response
                question_progress.progress((i + 1) / len(fixed_questions))
        
        # Store results for this file
        all_results[uploaded_file.name] = file_results
        
        # Update overall progress
        overall_progress.progress((file_idx + 1) / len(uploaded_files))
        
        # Clear memory between files
        pdf_text = None
        
    st.session_state.all_results = all_results
    
    # Create a DataFrame for Excel export
    excel_data = []
    
    for filename, results in all_results.items():
        for question, answer in results.items():
            excel_data.append({
                'Filename': filename,
                'Question': question,
                'Answer': answer
            })
    
    df = pd.DataFrame(excel_data)
    
    # Create Excel file with xlsxwriter
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write consolidated data to first sheet
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Write individual file results to separate sheets
        for filename, results in all_results.items():
            # Create a DataFrame for this file
            file_df = pd.DataFrame(list(results.items()), columns=['Question', 'Answer'])
            # Use a valid sheet name (max 31 chars, no special chars)
            sheet_name = os.path.splitext(filename)[0][:31]
            file_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Display results
    st.subheader("Extracted Data Summary")
    st.write(f"Processed {len(all_results)} PDF files")
    
    # Create tabs for each file's results
    tabs = st.tabs([f"File {i+1}: {name}" for i, name in enumerate(all_results.keys())])
    
    for i, (filename, tab) in enumerate(zip(all_results.keys(), tabs)):
        with tab:
            st.write(f"**Results from {filename}:**")
            for question, answer in all_results[filename].items():
                st.write(f"**{question}**: {answer}")
    
    # Download button for Excel with xlsxwriter
    st.download_button(
        label="Download Excel file with all results",
        data=buffer.getvalue(),
        file_name="all_pdf_extracted_data.xlsx",
        mime="application/vnd.ms-excel"
    )

elif uploaded_files:
    st.info("Click 'Extract Data from All PDFs' to process the files and extract information.")
else:
    st.info("Please upload PDF files to start extraction.")
