import warnings
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from uuid import uuid4
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
openai.api_key = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings(action='ignore')

load1 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/OS.pdf')
load2 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/DL.pdf')
doc1 = load1.load()
doc2 = load2.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50, length_function = len, is_separator_regex = False)
splitted_doc1 = splitter.split_documents(doc1)
splitted_doc2 = splitter.split_documents(doc2)

document = splitted_doc1 + splitted_doc2
uuids = [str(uuid4()) for _ in range(len(document))]


