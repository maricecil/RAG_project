# -*- coding: utf-8 -*-

import os
import json
import time
import pathlib
import webbrowser
from PIL import Image
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

image = Image.open("로고1.png")
st.sidebar.image(image, use_container_width=True)

# 현재 날짜와 시간 변수 정의
current_day = datetime.now().strftime('%Y-%m-%d')
current_time = datetime.now().strftime('%H:%M:%S')

# 사이드바에 시간 정보와 학습 시간 표시
st.sidebar.markdown("### 시간 정보", unsafe_allow_html=True)

# 현재 날짜/시간 표시
col1, col2 = st.sidebar.columns(2)
col1.markdown('<p style="color:white; font-weight:bold;">현재 날짜</p>', unsafe_allow_html=True)
col2.markdown(f'<p style="color:white; font-weight:bold;">{current_day}</p>', unsafe_allow_html=True)

col3, col4 = st.sidebar.columns(2)
col3.markdown('<p style="color:white; font-weight:bold;">현재 시간</p>', unsafe_allow_html=True)
col4.markdown(f'<p style="color:white; font-weight:bold;">{current_time}</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")  # 구분선 추가

# 세션 상태 초기화 - 파일 상단에 위치
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}
if "user_start_times" not in st.session_state:
    st.session_state["user_start_times"] = {}

# 이후 user_start_times를 사용하는 코드는 위의 초기화가 완료된 후에 실행됨
if st.session_state["user_id"]:
    user_id = st.session_state["user_id"]
    if user_id not in st.session_state["user_start_times"]:
        st.session_state["user_start_times"][user_id] = datetime.now()
    
    elapsed_time = datetime.now() - st.session_state["user_start_times"][user_id]
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    col5, col6 = st.sidebar.columns(2)
    col5.markdown('<p style="color:white; font-weight:bold;">학습 시간</p>', unsafe_allow_html=True)
    col6.markdown(f'<p style="color:white; font-weight:bold;">{hours:02d}:{minutes:02d}:{seconds:02d}</p>', unsafe_allow_html=True)

# 기본 경로 설정
project_root = "C:/Users/user/Desktop/JSW"
questions_folder = os.path.join(project_root, "Question")

# PDF 파일 경로 설정
pdf_file_path = {
    "파이썬": os.path.join(project_root, "data", "python", "python.pdf"),
    "머신러닝": os.path.join(project_root, "data", "machine_learning", "machine_learning.pdf"),
    "딥러닝": os.path.join(project_root, "data", "deep_learning", "deep_learning.pdf"),
    "데이터구조": os.path.join(project_root, "data", "data_structure", "data_structure.pdf"),
    "운영체제": os.path.join(project_root, "data", "operating_system", "operating_system.pdf"),
    "네트워크": os.path.join(project_root, "data", "network", "network.pdf"),
    "통계": os.path.join(project_root, "data", "statistics", "statistics.pdf"),
    "알고리즘": os.path.join(project_root, "data", "algorithm", "algorithm.pdf"),
}

# 파일 매핑 오타 수정
topic_file_mapping = {
    "파이썬": "python_module.txt",
    "머신러닝": "machine_learning_module.txt",
    "딥러닝": "deep_learning_module.txt",
    "데이터구조": "data_structure_module.txt",
    "운영체제": "operating_system_module.txt",  # opearting -> operating 오타 수정
    "네트워크": "network_module.txt",
    "통계": "statistic_module.txt",
    "알고리즘": "algorithm_module.txt",
}

# 질문 파일 로드 함수 수정
with st.spinner(text="In progress"):
    time.sleep(3)

def load_questions(topic):
    file_name = topic_file_mapping.get(topic)
    if not file_name:
        st.error(f"'{topic}' 주제에 대한 파일 매핑을 찾을 수 없습니다.")
        return []
        
    questions_file = os.path.join(questions_folder, file_name)
    
    if not os.path.exists(questions_file):
        st.error(f"질문 파일을 찾을 수 없습니다: {questions_file}")
        return []
        
    try:
        with open(questions_file, "r", encoding="utf-8") as file:
            questions = [q.strip() for q in file.readlines() if q.strip()]
            return questions
    except Exception as e:
        st.error(f"파일 읽기 오류: {str(e)}")
        return []

# ChatHistory 클래스 수정
class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, content, role="assistant"):
        """새 메시지를 대화 기록에 추가합니다."""
        if role == "user":
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": "assistant", "content": content})

    def get_history(self):
        """대화 기록을 반환합니다."""
        return [(msg["role"], msg["content"]) for msg in self.messages]

    def clear_history(self):
        """대화 기록을 초기화합니다."""
        self.messages = []

# FAISS 인덱스 저장 경로 수정
faiss_index_folder = os.path.join(project_root, "faiss_indexes")
os.makedirs(faiss_index_folder, exist_ok=True)

# .env 파일 로드
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# API 키 형식 검증
if not api_key.startswith("sk-"):
    st.error("올바르지 않은 API 키 형식입니다.")
    st.stop()

# 사용자 데이터 경로
users_file_path = pathlib.Path("users.json")

# 사용자 데이터 로드 함수
def load_users():
    if users_file_path.exists():
        with open(users_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 로그인된 사용자 확인
if not st.session_state["user_id"]:
    st.error("로그인되지 않았습니다. 먼저 로그인해주세요.")
    st.stop()

users = load_users()
user_id = st.session_state["user_id"]

if user_id not in users:
    st.error("사용자 정보를 확인할 수 없습니다. 다시 로그인해주세요.")
    st.stop()

# <div> 사용: 텍스트를 감싸는 상자를 만들기 위해 사용.
# color: gray;: 텍스트 색상을 연회색으로 설정.
# background-color: black;: 배경 색상을 검은색으로 설정.
# padding: 10px;: 상자의 내부 여백 추가.
# border-radius: 5px;: 상자의 모서리를 둥글게 처리.
# text-align: center;: 텍스트를 중앙 정렬.
# unsafe_allow_html=True: HTML 태그가 적용되도록 설정.

st.markdown(
    f"""
    <div style=" 
        color: black; 
        font-weight: bold; 
        font-size: 20px; 
        background-color: lightgray; 
        padding: 10px; 
        border-radius: 5px; 
        text-align: center;
        margin-top: 10px;">
        환영합니다, {user_id}님!
    </div>
    """,
    unsafe_allow_html=True
)

# 주제 선택 (바로 selectbox만 표시)
selected_topic = st.sidebar.selectbox("주제를 선택하세요", list(pdf_file_path.keys()))
questions = load_questions(selected_topic)

# 주제별 대화 기록 관리 (초기화가 중복되지 않도록)
if selected_topic not in st.session_state.chat_history:
    st.session_state.chat_history[selected_topic] = ChatHistory()

chat_history = st.session_state.chat_history[selected_topic]

# FAISS 저장 및 로드 함수
def save_faiss_index(vectorstore, topic):
    sanitized_topic = topic_file_mapping.get(topic, topic).replace("_module.txt", "")
    index_path = os.path.join(faiss_index_folder, f"{sanitized_topic}_index")
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)

def load_faiss_index(topic, api_key):
    sanitized_topic = topic_file_mapping.get(topic, topic).replace("_module.txt", "")
    index_path = os.path.join(faiss_index_folder, f"{sanitized_topic}_index")
    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

# 문서 처리 및 백터 스토어 초기화
def process_pdf(selected_topic, api_key):
    pdf_path = pdf_file_path.get(selected_topic)
    if pdf_path and os.path.exists(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                vectorstore = FAISS.from_documents(texts, embeddings)
                save_faiss_index(vectorstore, selected_topic)
                return vectorstore
            else:
                st.warning(f"{selected_topic} PDF 파일에서 데이터를 추출하지 못했습니다.")
        except Exception as e:
            st.write()
    return None

try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = load_faiss_index(selected_topic, api_key)
    if not vectorstore:
        st.warning("FAISS 인덱스를 로드 중입니다. 잠시만 기다려 주세요.")
        vectorstore = process_pdf(selected_topic, api_key)
except Exception as e:
    st.error(f"벡터 저장소 초기화 오류: {str(e)}")
    st.stop()

# 질문 로드
with st.spinner(text="In progress"):
    time.sleep(3)
questions = load_questions(selected_topic)

# FAISS 로드 및 생성
if not vectorstore:
    pdf_path = pdf_file_path.get(selected_topic)
    if pdf_path and os.path.exists(pdf_path):
        texts = process_pdf(pdf_path, api_key)

        if texts:
            vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=api_key))
            save_faiss_index(vectorstore, selected_topic)
        else:
            st.write()
    else:
        st.error(f"{selected_topic} 주제의 PDF 파일을 찾을 수 없습니다.")

# 선택하지 않음일 경우 안내 메시지
if selected_topic == "선택하지 않음":
    st.warning("PDF 파일을 선택하지 않았습니다. 주제를 선택해 주세요.")
    st.stop()

# 질문 선택
selected_question = st.sidebar.selectbox("질문 선택", questions)

# 메인 페이지 구성
st.title("자습실")

# 질문 출력 (모델 실행과 분리)
if selected_question:
    # 질문 중복 체크 (수정된 부분)
    existing_messages = chat_history.get_history()
    if not any(content == selected_question and role == "assistant" 
              for role, content in existing_messages):
        chat_history.add_message(selected_question, role="assistant")

# 질문 및 대화 기록 유지
for message in chat_history.get_history():
    try:
        role, content = message  # 튜플 언패킹
        if role == "assistant":
            st.chat_message("assistant").markdown(content)
        elif role == "user":
            st.chat_message("user").markdown(content)
    except Exception as e:
        st.error(f"메시지 포맷팅 오류: {str(e)}")
        continue

# 사용자 입력 처리
user_input = st.chat_input("질문이나 답변을 입력하세요...", key="user_chat_input")

# 모델 실행: 사용자 입력이 있을 경우에만
if user_input:
    chat_history.add_message(user_input, role="user")
    st.chat_message("user").markdown(user_input)

    try:
        # ChatOpenAI 객체 초기화
        chat_model = ChatOpenAI(
            model_name="gpt-4o",        # model -> model_name으로 수정
            openai_api_key=api_key,    # api_key -> openai_api_key로 수정
            temperature=0.7
        )

        if not vectorstore:
            st.error("FAISS 인덱스가 로드되지 않았습니다. PDF 파일 경로 또는 데이터를 확인하세요.")
            st.stop()
        
        with st.spinner(text="running"):
            time.sleep(5)

        # ConversationalRetrievalChain 초기화 및 실행
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
            ),
            return_source_documents=False,
            memory=None,  # 명시적으로 memory 설정
            verbose=True  # 디버깅을 위한 로그 활성화
        )

        # 사용자 입력과 대화 기록을 기반으로 프롬프트 생성 및 체인 실행
        formatted_history = []
        for role, content in chat_history.get_history():
            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_history.append(AIMessage(content=content))

        prompt = f"""
        면접 질문: {selected_question}
        사용자 응답: {user_input}
        You are an AI assistant tasked with answering questions using the documentation provided. 
        Always prioritize document content before answering questions.
        You must remember all previous chats, and your answers should match the language of the question, even if the content is in English.
        Address the other person as 'Candidate'.
        """

        response = chain(
            {
                "question": prompt,
                "chat_history": formatted_history,
            }
        )
        ai_response = response.get("answer", "응답을 생성하지 못했습니다.")

    except Exception as e:
        ai_response = f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}"
        st.error(ai_response)

    finally:
        # AI 응답을 추가
        chat_history.add_message(ai_response, role="assistant")
        st.chat_message("assistant").markdown(ai_response)
