import warnings
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from uuid import uuid4
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from nltk.corpus import stopwords
import re
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings(action='ignore')

load1 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/OS.pdf')
load2 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/DL.pdf')
try:
    doc1 = load1.load()
    doc2 = load2.load()
except Exception as e:
        print(f'PDF 파일에 에러가 있습니다: {e}')


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
splitted_doc1 = splitter.split_documents(doc1)
splitted_doc2 = splitter.split_documents(doc2)

document = splitted_doc1 + splitted_doc2
uuids = [str(uuid4()) for _ in range(len(document))]

embeddings = OpenAIEmbeddings()
faiss = FAISS.from_documents(documents = document, embedding = embeddings, ids = uuids)

query = 'Deep Learning관련 면접질문과 답변을 pdf파일로 준비해왔어. 여기 올리면 진위여부를 분별해 줄 수 있어??'
parser = StrOutputParser()
result = faiss.similarity_search(query, k = 5)
class datapreprocessing:
    def __init__(self, result):
        self.result = result
        
    def clean_text(self): 
        text = ' '.join([doc.page_content for doc in self.result])
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        sentences = text.split('. ')
        clean_text = '. '.join([sentence.strip() for sentence in sentences if sentence])
        return clean_text
        
data_instance = datapreprocessing(result) 
return_instance = data_instance.clean_text() 
# class CleanText(result):
# def stopwords(result):
#     stop_words = set(stopwords.words('korean'))
#     clean_text = [word for word in result if word not in stop_words]
#     return ' '.join(clean_text)

# def clean_text(result):
#     return re.sub(r'[^가-힣\s]', '', result)

client = OpenAI()
completion = client.chat.completions.create(
    model = 'gpt-4o',
    messages = [
        {'role' : 'system', 'content' : "지금부터 내가 준 텍스트를 변형, 추가 혹은 다른 단어, 그리고 마음대로 다르게 변경하면 절대 안되는데, 그냥 읽기 좋게 바꿔줘"} ,
        {'role' : 'user', 'content': return_instance}
    ],
    temperature = 0,
    max_tokens= 5000
    )
final_answer = completion.choices[0].message.content
print(final_answer)