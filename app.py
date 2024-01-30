from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# LangChain imports
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

# Load environment variable
api_key = os.getenv('OPENAI_API_KEY')

# LangChain setup
embeddings = OpenAIEmbeddings(api_key=api_key)
loader = DirectoryLoader('data', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Indexing documents
docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=api_key), 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)

@app.route('/query', methods=['POST'])
def handle_query():
    question = request.json['question']
    answer = qa.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
