# 📄 Document RAG System

A simple yet powerful Retrieval-Augmented Generation (RAG) application for personal document analysis. This Streamlit-based tool allows you to upload multiple documents, process them individually using LlamaIndex, and receive intelligent answers to predefined questions — all powered by OpenAI models.

---

## 💡 Powered by LlamaIndex

This application leverages [**LlamaIndex**](https://llamaindex.ai/) to seamlessly manage the document ingestion, indexing, and query workflows.

### 🔍 How LlamaIndex Is Used

- **Document Loading**: Uses `SimpleDirectoryReader` to load and parse various file formats.
- **Vector Indexing**: Documents are indexed using `VectorStoreIndex` with OpenAI's embedding model.
- **Query Engine**: The app constructs a query engine that finds the top-k most relevant chunks to answer each question.
- **Abstraction Layer**: LlamaIndex abstracts away the complexity of retrieval and lets you focus on asking meaningful questions.

> LlamaIndex is the backbone that connects raw documents to powerful LLM responses — streamlining the RAG workflow into a few lines of code.

---

## 💰 Why This RAG System is Cost-Effective

This app uses the **`gpt-4o-mini`** language model and **`text-embedding-3-small`** embedding model from OpenAI, both of which are **significantly cheaper** than full-size models while maintaining excellent performance for document-level tasks.

- 💸 **Low Token Costs**: Smaller models = lower API usage costs.
- 🧠 **Efficient Queries**: Only top 3 relevant chunks are queried per document.

Perfect for personal projects, hobby usage, or budget-conscious teams.

---

## 🚀 Features

- 🔄 **Batch Document Upload**: Upload and process multiple documents at once.
- 🧠 **LLM Integration**: Utilizes OpenAI's `gpt-4o-mini` for high-quality answers.
- 📋 **Customizable Questions**: Asks a fixed set of key questions per document.
- 📊 **Excel Export**: Automatically compiles answers into an Excel report for easy review.
- 🎛️ **Streamlit UI**: Clean and responsive interface built with Streamlit.

---

## 📁 Supported File Types

- PDF
- TXT
- DOCX
- CSV
- XLSX

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/VenkataSakethDakuri/RAG_Llamaindex
   cd RAG_Llamaindex
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

---

## 🧪 Usage

Run the application locally:

```bash
streamlit run app.py
```

You will be prompted to upload documents

---

## 📤 Output

After processing, results are:

- Displayed in the browser per document
- Compiled into an Excel file (`document_answers_<timestamp>.xlsx`)
- Downloadable directly through the interface

---

## 🙌 Contributing

Feel free to fork this repo and enhance it — such as adding new question sets, supporting more file types, or integrating with other vector stores.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📚 Acknowledgements

- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
