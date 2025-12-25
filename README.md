📄 Personal File Intelligent Chat Bot
An interactive chatbot built with Streamlit and LangChain that allows you to upload PDF files, extract their content, and ask intelligent questions about them.
The bot supports two backends for embeddings and LLM responses:
- OpenAI (GPT + embeddings)
- Google Gemini (Generative AI embeddings + Gemini chat models)

🚀 Features
- Upload PDF files and automatically extract text.
- Split text into manageable chunks for efficient processing.
- Generate embeddings using either OpenAI or Gemini.
- Store embeddings in a FAISS vector database for fast similarity search.
- Ask natural language questions about your document.
- Get contextual answers powered by your chosen LLM.
- Built with Streamlit for a simple and interactive UI.
- Includes caching logic to avoid hitting API limits unnecessarily.

🛠 Tech Stack
- Python 3.9+
- Streamlit (UI)
- PyPDF2 (PDF parsing)
- LangChain (chains, embeddings, vector stores)
- FAISS (vector similarity search)
- OpenAI / Gemini APIs (LLM + embeddings)

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME


2. Install Dependencies
pip install -r requirements.txt


3. Set API Keys
You’ll need either an OpenAI API key or a Gemini API key.
- For OpenAI:
export OPENAI_API_KEY="your_openai_key_here"
- For Gemini:
export GOOGLE_API_KEY="your_gemini_key_here"


⚠️ Keep your keys secure. Do not commit them to GitHub.

4. Run the App
streamlit run app.py



📂 Project Structure
├── ChatBot_openai.py     # OpenAI version
├── ChatBot_gemini.py     # Gemini version
├── requirements.txt  # Dependencies
├── README.md         # Project documentation



🔑 Usage
- Launch the app with Streamlit.
- Upload a PDF file from the sidebar.
- Type your query in the text box.
- The bot will:
- Perform similarity search on document chunks.
- Use the selected LLM (OpenAI or Gemini) to generate an answer.

🧩 Example Queries
- “Summarize chapter 2 of this PDF.”
- “What are the key points about AI mentioned in this document?”
- “Explain the financial data trends in this report.”

📌 Notes
- Gemini embeddings may require billing enabled in Google Cloud.
- Use caching to avoid hitting API limits.
- You can switch between OpenAI and Gemini by running the respective script.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.


