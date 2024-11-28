import warnings
import json
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import faiss
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import json
from datetime import datetime
from ML_module import ml_questions
from DL_module import dl_questions
from algorithm_module import algorithm_question
from OS_module import os_questions
from Network_module import network_questions
from DS_module import ds_questions
from python_module import python_questions
from statistic_module import statistic_questions
import time
import torch
#환경변수에서 api가 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY1")
#경고 무시 코드 입니다.(굳이 필요 없습니다. 경고창이 거슬리면 넣으시면 됩니다.)
warnings.filterwarnings(action='ignore')
#이 llm은 MultiQueryRetriever을 쓸 때, llm에 넣을 llm입니다.
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=500,
)
#client는 밑에서 쓰일 OpenAI입니다.
client = OpenAI()
model = ChatOpenAI(model="gpt-4")
#여기서 미리 텍스트를 청크로 나눌 모델을 서언합니다
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
#임베딩 방법을 선택합니다.
embeddings = OpenAIEmbeddings()

# 여기서 전부 주석 처리를 하는 이유는 여기서 먼저 다운을 받으면 더 이상 로드 할 필요가 없기 때문입니다.
# 빈 리스트를 하나 만듭니다
# file_list = []
# 여기서 pdf파일을 불러올 경로를 적습니다.
# local = [
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/OS.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/DL.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/ML.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/Python.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/Algo.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/DS.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/NW.pdf'),
#     PyPDFLoader('C:/Users/kevinkim/Downloads/project/Stat.pdf')
# ]
# 혹시 모르니 try except을 써줍니다
# try:
# 이 줄에서 먼저 local에 있는 pdf파일들을  하나씩 loader에 넣고
#     for loader in local:
# loader.load()를 통해 doc에 넣습니다.
#         docs = loader.load()
# 그리고 load()를 통해 불러온 값들을 하나씩 .extend를 통해 file_list에 넣습니다.
#         file_list.extend(splitter.split_documents(docs))
# 만약 불러올 수 없으면 에러가 뜹니다
# except Exception as e:
#         print(f'PDF 파일에 에러가 있습니다: {e}')
# 이 부분은 uuid를 만드는 과정이니 생략해도 무방합니다. 하지만 uuid는 컴퓨터가 파일을 더 읽기 쉽게 만듭니다.(지금은 설명 생략인데 혹시 궁금하시면 물어봐주세요)
# try:
#     uuids = [str(uuid4()) for _ in range(len(file_list))]
# 이제 불러온 file_list를 documents에 넣고, embedding, ids에 uuids를 넣어서 백터 스토어를 생성합니다
#     vector_store = FAISS.from_documents(documents=file_list, embedding=embeddings, ids = uuids)
# 그리고 .save_local("vector_store.bin")을 통해 vector_store.bin이라는 곳에 저장합니다(.bin은 그냥 가볍에 어떤 것을 저장하는 곳이라고 생각해주세요)
#     vector_store.save_local("vector_store.bin")
#     print("vector_store.bin 저장 완료")
# except Exception as e:
#     print(f"FAISS 저장 에러: {e}")
#그리고 이제 load_local을 써서 vector_store.bin에서 값을 불러온 후 다시 한번 백터 스토어를 생성합니다.
vector_store = FAISS.load_local("vector_store.bin", embeddings=embeddings, allow_dangerous_deserialization=True)
#이 함수는 값을 받으면 전처리를 해주는 코드입니다.
class datapreprocessing:
    def __init__(self, result):
        self.result = result
        
    def clean_text(self):
        text = re.sub(r'\s+', ' ', self.result)
        text = text.strip()
        
        sentences = text.split('. ')
        clean_text = '. '.join([sentence.strip() for sentence in sentences if sentence])
        return clean_text
#여기서 리스트리서 설정인데, 첫번째는 MultiQueryRetriever를 설정하기 위한 리트리버입니다.
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': 5})
#이제 여기서 멀티쿼라를 설정해주고, 위에서 설정해 놓았던 llm과 리트리버를 다시 넣습니다.
retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

#여기서 이제 프롬프트를 설정합니다.
contextual_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an AI assistant tasked with answering questions using the provided document. Always prioritize the document content for answering the question. "
    "BUT, you really must rememver the previous chat and you can add some extra information using previous chats If the document lacks sufficient information."
    "you must remember all the previous chats. furthermore, even if the content is English, your response will match the language of the query."
        ),
    HumanMessagePromptTemplate.from_template("documents: {documents}, question: {question}")
])

#이 클래스는 프롬프트에 값을 알맞게 넣어주는 역할을 하는 함수입니다.
class documenttoprompt:
    def __init__(self, contextual_prompt):
        self.contextual_prompt = contextual_prompt
    def invoke(self, input):
        if isinstance(input, list):
            context_doc = '\n'.join([doc.page_content for doc in input])
        else:
            context_doc = input
        
        formatted_prompt = self.contextual_prompt.format_messages(
            documents = context_doc,
            question = input.get('question', '')
        )
        return formatted_prompt
#이건 리트리버를 받아 값을 찾아서 반환하는 함수입니다.
class Retriever:
    def __init__(self, retriever):
        self.retriever = retriever
    def invoke(self,input):
        if isinstance(input, dict):
            query = input.get('question', "") 
        else:
            query = input
        response_doc = self.retriever.get_relevant_documents(query)
        return response_doc
#이제 레그에 맞게 각각 알맞은 값들을 넣어줍니다.
rag_chain = {
    'context' : Retriever(retriever),
    'prompt' : documenttoprompt(contextual_prompt),
    'llm' : model
}
#기본적으로 바꿔주세요! 이제 여기서 각 프롬프트와 리설트를 가져올 경로를 적습니다.
prompt_dir = r'C:/Users/kevinkim/Desktop/prompt/'
result_dir = r'C:/Users/kevinkim/Desktop/results/'

os.makedirs(prompt_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
#이건 쿼리를 적으면 크 쿼리의 느낌표나 물음표를 제거하기 위한 전처리 작업 함수입니다.
def filename_preprocessing(query):
    query = re.sub(r'[^\w\s]', '', query)
    return query
#이건 지금 이 코드에서 쓰이지 않습니다. 혹시 쓸 필요 있으시면 쓰셔도 무방하며, 삭제해도 전혀 지장이 없습니다.
def load_prompt(prompt_name):
    """프롬프트 파일 읽기"""
    prompt_path = os.path.join(prompt_dir, prompt_name)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"프롬프트 파일 {prompt_name}이(가) 존재하지 않습니다.")
    
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()
#이 코드는 질문을 저장하기 위한 함수입니다.
def save_prompt(query):
    query = filename_preprocessing(query)
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    prompt_file_name = f'{query}_{date_time}.txt'
    prompt_path = os.path.join(prompt_dir, prompt_file_name)

    # 디버깅: 저장하려는 query 출력
    print(f"Saving prompt: {query}_{date_time}.txt")
    
    with open(prompt_path, 'a', encoding='utf-8') as file: #a - write와 append를 동시에
        try:
            file.write(query)
            print(f"질문 저장 완료: {prompt_path}")
        except Exception as e:
            print(f'에러발생: {e}')
    print(f"저장된 질문: {query}")
#이 함수는 답변을 저장하기 위한 함수입니다.
def save_result(query, answer):
    query = filename_preprocessing(query)
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    result_file_name = f'{query}_{date_time}.txt'
    result_path = os.path.join(result_dir, result_file_name)
    
    # 디버깅: 저장하려는 answer 출력
    print(f"Saving answer: {query}_{date_time}.txt")
    
    with open(result_path, 'a', encoding='utf-8') as file:
        try:
            file.write(answer)
            print(f"결과 저장 완료: {result_path}")
        except Exception as e:
            print(f'에러발생: {e}')
            
def clean_query(query):
    # 영어, 한글, 숫자, 공백, 그리고 구두점만 허용하고, 그 외의 특수 문자나 시스템 명령어 제거
    query = re.sub(r'[^\w\s]', '', query)
    return query
chat_history = ChatMessageHistory()
set_llm_cache(InMemoryCache())
while True:
    #이건 그냥 목록표를 만든것입니다.
    print("""반갑습니다! 당신의 IT 면접을 도와드릴 AI 입니다! 무엇을 도와드릴까요?
          먼저 어떤 주제를 원하시나요?
          1. 딥러닝(Deep Learning)
          2. 머신러닝(Machine Learning)
          3. 파이썬(Python)
          4. 알고리즘(Algorithm)
          5. 데이터구조(Data Structure)
          6. 네트워크(Network)
          7. 통계학(statistics)
          8. 운영체제(Operating System)
          9. 과거 질문 리스트(list) 
          10. 나가기(끄기)
          원하는 주제의 숫자혹은 주제의 이름을 적어주세요!
          """)
    #이 부분도 똑같이 목록표입니다.
    top_page = input("원하는 주제를 골라주세요:")
    if top_page in ['10', 'quit',  'exit',  'out', '나가기']:
        print('즐거운 시간 되셨길 바랍니다!')
        break
    elif top_page.lower() in ['1', '딥러닝', '딥 러닝', 'deep learning', 'deeplearning']:
        print('딥러닝을 고르셨군요! 면접때 자주 나오는 딥러닝 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(dl_questions)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['2', '머신러닝', '머신 러닝', 'machine learning', 'machinelearning']:
        print('머신러닝을 고르셨군요! 면접때 자주 나오는 머신러닝 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(ml_questions)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['3', '파이썬', 'python']:
        print('파이썬을 고르셨군요! 면접때 자주 나오는 파이썬 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(python_questions)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['4', '알고리즘', 'algorithm']:
        print('알고리즘을 고르셨군요! 면접때 자주 나오는 알고리즘 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(algorithm_question)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['5', '데이터구조', 'data structure', 'datastructure']:
        print('데이터구조을 고르셨군요! 면접때 자주 나오는 데이터구조 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(ds_questions)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['6', '네트워크', 'network']:
        print('네트워크을 고르셨군요! 면접때 자주 나오는 네트워크 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(network_questions)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['7', '통계학', 'statistics', 'statistic']:
        print('통계학을 고르셨군요! 면접때 자주 나오는 통계학 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(statistic_questions)
        print('주제로 돌아갈면 back을 치세요!')
    elif top_page.lower() in ['8', '운영체제', 'operating system', 'operatingsystem']:
        print('운영체제을 고르셨군요! 면접때 자주 나오는 운영체제 문제들을 보여드리겠습니다. 좋은 시간 보내세요!' )
        print(os_questions)
        print('주제로 돌아갈면 back을 치세요! 혹은 프로그램을 끄실꺼면 나가기를 타입해주세요!')
        #목록읅 관리하고 삭제하기 위한 일에 들어가겠습니다.
    # 전체 대화 히스토리 확인
    elif top_page.lower() in ['9', 'list','show me the list','리스트를 보여줘', '리스트']:
        print('\n현재 대화 히스토리')
        #여기서 먼저 result를 저장해둔 디렉토리 경로를 가져옵니다.
        folder = os.listdir('C:/Users/kevinkim/Desktop/results/')
        #여기서 enumerate을 통해 index와 그에 따른 폴더들을 가져옵니다
        for index, folder_list in enumerate(folder, start = 1):
            #이렇게 되면 화면에 1. 이름 이런 식으로 나옵니다. 추후에 편집 가능합니다.
            print(f'질문: {index} 대답:{folder_list}')
        choose = input('혹시 다시 보고 싶은 답변이 있으시나요? 원하는 번호를 선택해 주세요!: ')
        #여기서 다시 한번 숫자를 선택할 수 있습니다. 
        for index, file_name in enumerate(folder, start = 1):
            new_file = os.path.join('C:/Users/kevinkim/Desktop/results/' + file_name) 
            #이번에는 만약 선택한 숫자가 인덱스에 포함되어 있다면,
            if index == int(choose):
                #이번에는 파일을 열어 그 안의 값들을 가져옵니다.
                with open(new_file, 'r', encoding= 'utf-8') as file:
                    file_content = file.read()
                    file_name = file_name.split('_')[0]
                    print(f"\n질문 : {file_name} \n대답 : {file_content}")
                break
        #이제는 삭제하는 방법입니다.
        decision = input('''답변이 마음에 드셨나요?
            1. 주제 다시 고르기
            2. 나가기(quit)
            3. 답변 삭제하기
            ''')
        if decision in ['1', '주제 다시 고르기']:
            print('주제 고르기로 넘어갑니다')
            continue
        
        elif decision in ['2', 'quit',  'exit',  'out', '나가기']:
            print('즐거운 시간 되셨길 바랍니다!')
            break
        #만약 삭제하기를 눌러, 아까처럼 목록을 보여주는 리스트를 만들고,
        elif decision in ['3', '답변 삭제하기', '답변삭제하기']:
            folder = os.listdir('C:/Users/kevinkim/Desktop/results/')
            for index, file_name in enumerate(folder, start = 1):
                print(f'질문: {index} 대답:{file_name}')
            choose_file = input('삭제할 파일을 선택하세요!')
            for index, file_name in enumerate(folder, start = 1):
                if index == int(choose_file):
                    #이번에도 똑같지만 밑에 .remove를 통해 지워줍니다.
                    new_file = os.path.join('C:/Users/kevinkim/Desktop/results/' + file_name)
                    file_name = file_name.split('_')[0]
                    os.remove(new_file)
                    print('해당 답변은 삭제되었습니다')
                break
    #이건 그냥 잘못 적었을 경우 심심해서 넣어본 기능이니 삭제해도 무방합니다..
    else:
        print('정확한 주제를 골라주세요!')
        time.sleep(2)
        print('프로세스 다시 가동 중...')
        time.sleep(1)
        print('프로세스 가동완료...')
        time.sleep(1)
        continue
    
    #이제는 쿼리를 받아볼 차례입니다.
    query = input('질문을 입력하세요 : ')
    #먼저 위에 filename이라는 함수를 사용해 쿼리의 전처치를 합니다.
    query = filename_preprocessing(query)
    #만약 쿼리를 잘못 적거나 back을 치면 다시 돌아가는 것을 적었습니다.
    if query.lower() in ['10', 'quit',  'exit',  'out', '나가기', '끄기']:
        print('이용해 주셔서 감사합니다.')
        break
    if query.lower() in ['back']:
        continue
    chat_history.add_user_message(query)

#이건 만약 같은 질문을 넣었을 경우, 방지하는 목표로 만들었습니다 이것 또한 삭제해도 무방합니다.
    duplicate_question = os.listdir('C:/Users/kevinkim/Desktop/prompt/')
    pattern = r'_\d{4}-\d{2}-d{2}_\d{2}_\d{2}$'
    duplicate_folder = []
    for list in duplicate_folder:
        preprocessing_file = re.sub(pattern, '', duplicate_question)
        preprocessing_file = [file.replace('.txt', '') for file in preprocessing_file if file.endswith('.txt')]
        duplicate_folder.append(preprocessing_file)
        for text in duplicate_folder:
            try:
                if query.lower() in text:
                    print('이미 물어본 질문입니다. 다른 질문을 해주세요')
                    continue
            except Exception as e:
                print(f'에러발생: {e}')
    #이제 적은 쿼리를 저장하는 함수를 씁니다.
    save_prompt(query)
    #이제 위에서 썼던 함수들을 하나씩 쓸 차례입니다.
    #rag_chain에 있는 값 값들을 불러, invoke를 시킴과 동시에 각 알맞는 값들을 넣어줍니다.
    first_query = rag_chain['context'].invoke({'question' : query})
    second_query = rag_chain['prompt'].invoke({
        'context' : first_query,
        'question' : query
    })
    third_query = rag_chain['llm'].invoke(second_query)
    llm_result = third_query.content
    forth_query = datapreprocessing(llm_result)
    last_query = forth_query.clean_text() 
    #마지막으로 최종적으로 나온 값을 한번 더 걸러주는 AI를 만들었습니다.
    completion = client.chat.completions.create(
        model = 'gpt-4o',
        messages = [
            {'role' : 'system', 'content' : "From now on, you must not modify, add, or replace any words in the text I provide, nor change it in any way. Your only task is to make it easier to read. However, if the user communicates in another language, translate your response into that language. It's essential to note that multilingual support is a must."
             "furthermore, even if the content is not in English, your response will match the language of the query."} ,
            {'role' : 'user', 'content': last_query},
            {'role' : 'assistant', 'content' : """
             for example,
             user: 딥러닝이란
             assistant: 딥러닝은 머신러닝의 한 종류로, 여러 층을 가진 인공신경망을 사용하여 학습을 수행하는 기술입니다. 이를 심층학습이라고도 부릅니다. 딥러닝의 가장 큰 특징은 기계가 자동으로 학습하려는 데이터에서 중요한 특징을  추출하여 학습하고, 이를 바탕으로 의사결정이나 예측 등을 수행한다는 점입니다. 이는 기존의 머신러닝과의 주요 차이점으로, 머신러닝에서는 데이터의 특징을 사람이 직접 분석하고 판단해야 했지만, 딥러닝에서는 이  작업을 기계가 자동으로 수행합니다. 딥러닝은 대규모 데이터에서 중요한 패턴 및 규칙을 학습하는 기술로 정의할 수 있습니다.
             user: What is deep learning
             assistant: Deep learning is a type of machine learning that uses artificial neural networks with multiple layers to perform learning, also known as deep learning. The most significant feature of deep learning is its ability to automatically extract important features from the data it aims to learn, which it uses for decision-making or predictions. This is a key difference from traditional machine learning, where humans must manually analyze and determine the features of the data. In contrast, deep learning performs this task automatically. Deep learning can be defined as a technology that learns important patterns and rules from large-scale data.
             user: ディープラーニングとは
             assistant: ディープラーニングは、複数の層を持つ人工ニューラルネットワークを使用して学習を行う機械学習の一種であり、深層学習とも呼ばれます。ディープラーニングの最大の特徴は、学習対象のデータから重要な特徴を自動的に抽出し、それに基づいて意思決定や予測を行う点です。これが従来の機械学習との大きな違いであり、機械学習ではデータの特徴を人間が直接分析して判断する必要がありましたが、ディープラーニングではその作業を機械が自動で行います。ディープラーニングは、大規模なデータから重要なパターンや規則を学習する技術と定義することができます。
             user: Le deep learning est
             assistant: Le deep learning est un type d'apprentissage automatique qui utilise des réseaux neuronaux artificiels à plusieurs couches pour effectuer l'apprentissage, également appelé apprentissage profond. La caractéristique principale du deep learning est sa capacité à extraire automatiquement des caractéristiques importantes des données qu'il cherche à apprendre, sur lesquelles il se base pour prendre des décisions ou faire des prédictions. C'est une différence clé par rapport à l'apprentissage automatique traditionnel, où l'humain doit analyser et déterminer manuellement les caractéristiques des données. En revanche, avec le deep learning, cette tâche est réalisée automatiquement par la machine. Le deep learning peut être défini comme une technologie permettant d'apprendre des modèles et des règles importantes à partir de données à grande échelle.
             """}
        ],
        temperature = 0
        )
    answers = completion.choices[0].message.content
    #이건 이제 나온 값을 저장하는 함수를 썼습니다.
    save_result(query, answers)
    #이건 이제 이전 대화를 기억하도록 add_message를 썼습니다.
    chat_history.add_message(answers)
    print(answers, flush = True)