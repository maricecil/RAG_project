import warnings
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from uuid import uuid4
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.output_parsers import StrOutputParser
openai.api_key = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings(action='ignore')

load1 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/OS.pdf')
load2 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/DL.pdf')
try:
    doc1 = load1.load()
    doc2 = load2.load()
except Exception as e:
        print(f'PDF 파일에 에러가 있습니다: {e}')


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100, length_function = len, is_separator_regex = False)
splitted_doc1 = splitter.split_documents(doc1)
splitted_doc2 = splitter.split_documents(doc2)

document = splitted_doc1 + splitted_doc2
uuids = [str(uuid4()) for _ in range(len(document))]

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
faiss = FAISS.from_documents(documents = document, embedding = embeddings, ids = uuids)

query = '딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?'
result = faiss.similarity_search(query, k = 3)

print(result)
