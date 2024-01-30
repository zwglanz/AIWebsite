from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI  # Make sure this import is from langchain_openai
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

# Load environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('The environment variable OPENAI_API_KEY must be set')

# LangChain setup
embeddings = OpenAIEmbeddings(api_key=api_key)
loader = DirectoryLoader('data', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Indexing documents
docsearch = Chroma.from_documents(texts, embeddings)

# Create RetrievalQA instance with a valid chain type
# Replace 'retrieval' with a value from the following list: ['stuff', 'map_reduce', 'refine', 'map_rerank']
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=api_key),
    chain_type="stuff",  # This is an example; you need to choose the correct one for your application
    retriever=docsearch.as_retriever()
)

@app.route('/query', methods=['POST'])
def handle_query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Missing 'question' in the request body"}), 400
    answer = qa.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=False)  # Change debug to False for production
