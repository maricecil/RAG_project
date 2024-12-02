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

st.markdown(
    """<style>
    .stApp {
        background-color: #222831;
        color: white; /* 기본 글자 색상을 흰색으로 설정 */
        height: 100vh;
        margin: 0;
    }
    .stImage img {
        border-radius: 15px;         
        border: 2px solid #00000; 
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5); 
        max-width: 100%;                      
        display: block;
        margin: auto;
        width: 800px;
        height: 250px;
    }
    .stSidebar {
        background-color: #31363F;
        width: 300px;
    }
    .stText{
        
    }
    .stSelectbox {
        background-color: #EBEAFF;
        border: 2px solid #00000;
        padding: 10px;
        border-radius: 5px;
    }

    /* 사용자 메시지 및 모델 응답 텍스트 색상 */
    .stMarkdown {
        color: white !important; /* st.chat_message에서 사용되는 텍스트 색상 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

image = Image.open("C:/Users/USER/REG_project/RAG_project/AI_logo.png")
st.sidebar.image(image, use_container_width=True)

# 시간 및 날짜 출력 (매번 렌더링 시 갱신)
current_date_space = st.sidebar.empty()  # 빈 공간 생성
current_time_space = st.sidebar.empty()  # 빈 공간 생성

# 현재 날짜와 시간 출력
current_day = datetime.now().strftime('%Y-%m-%d')  # 현재 날짜 포맷팅
current_time = datetime.now().strftime('%H:%M:%S')  # 현재 시간 포맷팅

current_date_space.markdown(
    f'<p style="color:white; font-weight:bold;">현재 날짜 - {current_day}</p>',
    unsafe_allow_html=True,
)
current_time_space.markdown(
    f'<p style="color:white; font-weight:bold;">현재 시간 - {current_time}</p>',
    unsafe_allow_html=True,
)

# PDF 경로 매핑
pdf_file_path = {
    "선택하지 않음": "data/none.pdf",
    "파이썬": "C:/Users/USER/REG_project/RAG_project/data/python/python.pdf",
    "머신러닝": "C:/Users/USER/REG_project/RAG_project/data/machine_learning/machine_learning.pdf",
    "딥러닝": "C:/Users/USER/REG_project/RAG_project/data/deep_learning/deep_learning.pdf",
    "데이터구조": "C:/Users/USER/REG_project/RAG_project/data/data_structure/data_structure.pdf",
    "운영체제": "C:/Users/USER/REG_project/RAG_project/data/operating_system/operating_system.pdf",
    "네트워크": "C:/Users/USER/REG_project/RAG_project/data/network/network.pdf",
    "통계": "C:/Users/USER/REG_project/RAG_project/data/statistics/statistics.pdf",
    "알고리즘": "C:/Users/USER/REG_project/RAG_project/data/algorithm/algorithm.pdf",
}

# 질문 파일 경로 매핑
questions_folder = "C:/Users/USER/REG_project/RAG_project/김준기님/"
topic_file_mapping = {
    "파이썬": "python_module.txt",
    "머신러닝": "ML_module.txt",
    "딥러닝": "DL_module.txt",
    "데이터구조": "DS_module.txt",
    "운영체제": "OS_module.txt",
    "네트워크": "Network_module.txt",
    "통계": "statistic_module.txt",
    "알고리즘": "algorithm_module.txt",
}

# ChatHistory 클래스
class ChatHistory:
    def __init__(self):
        self.messages = []  # 대화 기록 저장

    def add_message(self, content, role="assistant"):
        """새 메시지를 대화 기록에 추가합니다."""
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        """대화 기록을 반환합니다."""
        return self.messages

    def clear_history(self):
        """대화 기록을 초기화합니다."""
        self.messages = []

# FAISS 인덱스 저장 경로
faiss_index_folder = "C:/Users/USER/REG_project/RAG_project/faiss_indexes"
os.makedirs(faiss_index_folder, exist_ok=True)

# .env 파일 로드
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")

# 사용자 데이터 경로
users_file_path = pathlib.Path("users.json")

# 사용자 데이터 로드 함수
def load_users():
    if users_file_path.exists():
        with open(users_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 세션 상태 초기화
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None  # 초기값 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 초기값 설정
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}  # 카테고리별 대화 기록

# session_state 초기화
# if "initialized" not in st.session_state:
#     st.session_state["initialized"] = True
#     st.session_state["chat_history"] = {}
#     st.session_state["messages"] = []
#     st.session_state["user_id"] = None

# 로그인된 사용자 확인
if not st.session_state["user_id"]:
    st.error("로그인되지 않았습니다. 먼저 로그인해주세요.")
    st.stop()

users = load_users()
current_user_id = st.session_state["user_id"]

if current_user_id not in users:
    st.error("사용자 정보를 확인할 수 없습니다. 다시 로그인해주세요.")
    st.stop()

st.success(f"환영합니다, {current_user_id}님!")

# 주제 선택
selected_topic = st.sidebar.selectbox("주제를 선택하세요", list(pdf_file_path.keys()))

# 주제별 대화 기록 관리 (초기화가 중복되지 않도록)
if selected_topic not in st.session_state.chat_history:
    st.session_state.chat_history[selected_topic] = ChatHistory()

chat_history = st.session_state.chat_history[selected_topic]

# 질문 파일 로드 함수
def load_questions(topic):
    file_name = topic_file_mapping.get(topic)
    if not file_name:
        return []
    questions_file = os.path.join(questions_folder, file_name)
    if not os.path.exists(questions_file):
        return []
    with open(questions_file, "r", encoding="utf-8") as file:
        return [q.strip() for q in file.readlines() if q.strip()]

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

vectorstore = load_faiss_index(selected_topic, api_key)

if not vectorstore:
    vectorstore = process_pdf(selected_topic, api_key)

if not vectorstore:
    st.write()
    st.stop()

# 질문 로드
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
st.title("AI 면접 튜터링")

# 질문 출력 (모델 실행과 분리)
if selected_question:
    # 질문을 화면에 출력 (단, 중복 출력 방지)
    if not any(msg["content"] == selected_question and msg["role"] == "assistant" for msg in chat_history.get_history()):
        chat_history.add_message(selected_question, role="assistant")

# 질문 및 대화 기록 유지
for message in chat_history.get_history():
    if message["role"] == "assistant":
        st.chat_message("assistant").markdown(f"**질문:** {message['content']}")
    elif message["role"] == "user":
        st.chat_message("user").markdown(message["content"])

# 사용자 입력 처리
user_input = st.chat_input("질문이나 답변을 입력하세요...", key="user_chat_input")

# 모델 실행: 사용자 입력이 있을 경우에만
if user_input:
    chat_history.add_message(user_input, role="user")
    st.chat_message("user").markdown(user_input)

    try:
        # ChatOpenAI 객체 초기화
        chat_model = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,
            temperature=0.7
        )

        if not vectorstore:
            st.error("FAISS 인덱스가 로드되지 않았습니다. PDF 파일 경로 또는 데이터를 확인하세요.")
            st.stop()
        
        # ConversationalRetrievalChain 초기화 및 실행
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=False,
        )

        # 사용자 입력과 대화 기록을 기반으로 프롬프트 생성 및 체인 실행
        formatted_history = []
        for message in chat_history.get_history():
            if message["role"] == "user":
                formatted_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))

        prompt = f"""
        면접 질문: {selected_question}
        사용자 응답: {user_input}
        You are an AI assistant tasked with answering questions using the documentation provided. 
        Always prioritize document content before answering questions.
        You must remember all previous chats, and your answers should match the language of the question, even if the content is in English.
        Address the other person as 'Candidate'.

        Provide the answer in the following order:
        1. Feedback: Specific feedback on the user's response.
        2. Suggestions: How to improve the answer.
        3. Additional comments: Additional tutoring on the topic.
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

