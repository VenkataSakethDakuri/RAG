import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import sys

from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)

Settings.llm = llm
Settings.embed_model = embed_model


data_dir = "C:\\Users\\DELL\\OneDrive\\Desktop\\VS CODE\\RAG\\data"
# if not os.path.exists(data_dir):
#     print(f"Error: Data directory does not exist: {data_dir}")
#     print("Please create this directory and add PDF files to it.")
#     sys.exit(1)


# pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
# if not pdf_files:
#     print(f"Error: No PDF files found in {data_dir}")
#     print("Please add PDF files to this directory.")
#     sys.exit(1)
# else:
#     print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

try:
    documents = SimpleDirectoryReader(
        input_dir=data_dir
    ).load_data()

    print(f"Loaded {len(documents)} document(s)")

    index = VectorStoreIndex.from_documents(documents)

    
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Retrieve top 3 most relevant chunks
        response_mode="compact"  # Options: "refine", "compact", "tree_summarize", compact has least llm calls 
    )

#BM25 + Vector Search: Combines traditional keyword-based retrieval (BM25) with embedding similarity search to capture both semantic meaning and exact keyword matches.
#Minimum Similarity Threshold: Instead of using a fixed top-k, you can filter retrieved nodes based on a minimum similarity score.

    def run_rag_chat():
        print("PDF RAG System with GPT-4o mini is ready! Type 'exit' to quit.")
        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'exit':
                break
            
            response = query_engine.query(query)
            print(f"\nResponse: {response}")

    if __name__ == "__main__":
        run_rag_chat()
except Exception as e:
    print(f"Error loading documents: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install pypdf pillow pdfminer.six")







#text embeddings 3 small : $0.02 per 1 million tokens (or $0.00002 per 1,000 tokens)

# #gpt 4o mini:
# Input tokens: $0.15 per 1 million tokens (15 cents per million)
# Output tokens: $0.60 per 1 million tokens (60 cents per million)
# Cached input: $0.075 per 1 million tokens