
import warnings
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from uuid import uuid4
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import faiss
from nltk.corpus import stopwords
import re
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
openai.api_key = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings(action='ignore')

load1 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/OS.pdf')
load2 = PyPDFLoader('C:/Users/kevinkim/Downloads/project/DL.pdf')
try:
    #doc1 = [page.extract_text() for page in load1.pages]
    #doc2 = [page.extract_text() for page in load2.pages]
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

# query =  input('어떤 질문을 원하시나요?: ')
parser = StrOutputParser()
# result = faiss.similarity_search(query, k = 5)
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

# data_instance = datapreprocessing(result) 
# return_instance = data_instance.clean_text() 

# class CleanText(result):
# def stopwords(result):
#     stop_words = set(stopwords.words('korean'))
#     clean_text = [word for word in result if word not in stop_words]
#     return ' '.join(clean_text)

# def clean_text(result):
#     return re.sub(r'[^가-힣\s]', '', result)

vector_store = FAISS.from_documents(documents=document, embedding=embeddings)
retriver = vector_store.as_retriever(search_type = 'simiarlity', search_kwargs = {'k': 5})

contextual_prompt = ChatPromptTemplate([
    {'system', 'Answer the questions using only the following document. you can add some extra information if needed'},
    {'user', 'documents: {documents}, question: {question}'}
])
class documenttoprompt:
    def __init__(self, contextual_prompt):
        self.contextual_prompt = contextual_prompt
    def invoke(self, input):
        if isinstance(input, list):
            context_doc = '\n'.join(doc.page_content for doc in input)
        else:
            context_doc = input
        
        formatted_prompt = self.contextual_prompt.format_messages(
            documents = context_doc,
            question = input.get('question', '')
        )
        return formatted_prompt


# client = OpenAI()
# completion = client.chat.completions.create(
#     model = 'gpt-4o',
#     messages = [
#         {'role' : 'system', 'content' : "지금부터 내가 준 텍스트를 변형, 추가 혹은 다른 단어로 교체, 그리고 마음대로 다르게 변경하면 절대 안되는데, 그냥 읽기 좋게 바꿔줘"} ,
#         {'role' : 'user', 'content': return_instance}
#     ],
#     temperature = 0,
#     max_tokens= 1000
#     )
# answer1 = completion.choices[0].message.content

# completion2 = client.chat.completions.create(
#     model = 'gpt-4o',
#     messages = [
#         {'role' : 'system', 'content' : 'translate the given content into {language}'},
#         {'role' : 'user', 'content': return_instance}
#     ],
#     temperature = 0,
#     max_tokens= 1000
#     )

# def language(language_selection):
#     if language_selection == 1 or '일본어' or 'Japanses':
#         translated_sentence = completion2.format(language = 'Japanses')
#     elif language_selection == 1 or '영어' or 'English':
#         translated_sentence = completion2.format(language = 'English')
#     elif language_selection == 1 or '한국어' or 'Korean':
#         translated_sentence = completion2.format(language = 'Korean')
#     elif language_selection == 1 or '프랑스어' or 'French':
#         translated_sentence = completion2.format(language = 'French')
#     elif language_selection == 1 or '독일어' or 'Germany':
#         translated_sentence = completion2.format(language = 'Germany')
#     elif language_selection == 1 or '러시아어' or 'Russian':
#         translated_sentence = completion2.format(language = 'Russian')
#     elif language_selection == 1 or '스페인어' or 'Spanish':
#         translated_sentence = completion2.format(language = 'Spanish')
#     else:
#         translated_sentence = completion2.format(language = language_selection)
#         return translated_sentence
    
# while True:
#     language_selection = input('''언어를 선택하세요:
#                            1. 일본어(Japanses)
#                            2. 영어(English)
#                            3. 한국어(Korean)
#                            4. 프랑스어(French)
#                            5. 독일어(Germany)
#                            6. 러시아어(Russian)
#                            7. 스페인어(Spanish)
#                            혹은 원하는 언어를 적어주세요: ''')
#     try:
#         print(language(language_selection))
#     except Exception as e:
#         print('죄송합니다. 작성하신 언어는 존재하지 않거나, 지원하지 않습니다. 언어를 다시 확인해 주세요')