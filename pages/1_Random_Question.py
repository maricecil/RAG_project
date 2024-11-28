import streamlit as st # 웹 애플리케이션을 만드는 라이브러리
import numpy as np
import random # 무작위 선택을 위한 라이브러리
import PyPDF2 # PDF 파일을 다루는 라이브러리
import io # 입출력 작업을 위한 라이브러리
import re # 텍스트 패턴을 찾는 라이브러리
import os # 파일/폴더 관리를 위한 라이브러리
import pathlib
from dotenv import load_dotenv # 환경변수를 불러오는 라이브러리
from langchain_core.messages.chat import ChatMessage # AI 채팅 메시지를 다루는 도구
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI의 AI 모델을 사용하기 위한 도구
from langchain_core.output_parsers import StrOutputParser # AI 출력을 문자열로 변환하는 도구
from langchain_teddynote.prompts import load_prompt # AI 지시사항을 불러오는 도구
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 텍스트를 나누는 도구
from langchain_community.vectorstores import FAISS # 텍스트를 검색하기 쉽게 저장하는 도구
from langchain_core.runnables import RunnablePassthrough # AI 처리 과정을 연결하는 도구
from langchain_teddynote import logging # 작업 기록을 남기는 도구
from langchain_community.document_loaders import PyPDFDirectoryLoader # PDF 파일을 읽는 도구
from langchain.prompts import PromptTemplate # AI 지시사항 템플릿을 만드는 도구
from streamlit_option_menu import option_menu
from PIL import Image
from utils import switch_page

st.set_page_config(page_title="무작위 질문")

# 사이드바 구성
with st.sidebar:
        st.markdown('AI 면접관')
        st.selectbox('#### 분야 선택', ['Python', 'Machine_Learning', 'Deep_Learning', 'Network', 'Statistics', 
                                     'Operating_System', 'Data_Structure', 'Algorithm'])
        
# 각 면접 주제(카테고리)
CATEGORY_MAPPING = {
    "파이썬": "python",
    "머신러닝": "machine_learning",
    "딥러닝": "deep_learning",
    "데이터구조": "data_structure",
    "운영체제": "operating_system",
    "네트워크": "network",
    "통계": "statistics",
    "알고리즘": "algorithm"
}

# 각 카테고리별로 어떤 PDF 파일을 읽을지 정하는 사전
PDF_FILE_MAPPING = {
    "파이썬": "C:/Users/user/RAG_project/JSW/QnA/python.pdf",
    "머신러닝": "C:/Users/user/RAG_project/JSW/QnA/machine_learning.pdf",
    "딥러닝": "C:/Users/user/RAG_project/JSW/QnA/deep_learning.pdf",
    "데이터구조": "C:/Users/user/RAG_project/JSW/QnA/data_structure.pdf",
    "운영체제": "C:/Users/user/RAG_project/JSW/QnA/operating_system.pdf",
    "네트워크": "C:/Users/user/RAG_project/JSW/QnA/network.pdf",
    "통계": "C:/Users/user/RAG_project/JSW/QnA/statistics.pdf",
    "알고리즘": "C:/Users/user/RAG_project/JSW/QnA/algorithm.pdf"
}

# AI에게 줄 지시사항(프롬프트)을 파일에서 불러옵니다
feedback_prompt = load_prompt("C:/Users/user/RAG_project/JSW/prompt/feedback_prompt.Yaml")  # 피드백용 지시사항
question_prompt = load_prompt("C:/Users/user/RAG_project/JSW/prompt/question_prompt.Yaml")  # 질문용 지시사항

# OpenAI API 키 등의 중요 정보를 환경변수에서 불러옵니다
load_dotenv()

# 필요한 폴더들이 없다면 새로 만듭니다
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 프로젝트 이름을 기록해 효율적인 관리를 수행
logging.langsmith("[Project] PDF_RAG")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [] #초기화

# 대화 내용을 저장할 변수들을 초기화합니다
if "messages" not in st.session_state:  # 이전 대화가 없다면
    st.session_state["messages"] = []   # 새로운 대화 목록을 만듭니다
if "chain" not in st.session_state:     # AI 처리 과정이 없다면
    st.session_state["chain"] = None    # 새로 만들 준비를 합니다
if "selected_category" not in st.session_state:  # 선택된 주제가 없다면
    st.session_state["selected_category"] = None # 새로 선택할 준비를 합니다

# 이미 사용한 질문들을 기억하기 위한 변수를 초기화합니다
if "used_topics" not in st.session_state:
    st.session_state["used_topics"] = set()
    
# PDF 파일을 읽어서 AI가 이해할 수 있는 형태로 바꾸는 함수
@st.cache_resource(show_spinner="면접질문을 준비하는 중입니다...")
def embed_files_from_directory(directory_path):
    try:
        # PDF 파일이 있는지 확인합니다
        if not os.path.exists(directory_path):
            st.error(f"디렉토리가 존재하지 않습니다: {directory_path}")
            return None

        # PDF 파일 목록을 가져옵니다
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        if not pdf_files:
            st.error(f"디렉토리에 PDF 파일이 없습니다: {directory_path}")
            return None

        # PDF 파일들을 읽어옵니다
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()

        if not docs:
            st.error("PDF 파일을 로드할 수 없습니다. 디렉토리를 확인하세요.")
            return None

# 긴 문서를 작은 조각으로 나눕니다
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)

        if not split_documents:
            st.error("문서를 분할할 수 없습니다. PDF 파일의 내용을 확인하세요.")
            return None

        # 텍스트를 AI가 이해할 수 있는 형태(임베딩)로 변환합니다
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"PDF 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return None


st.markdown("## 무작위 질문")

st.info("""
        📚이 세션에서는 AI 면접관이 사용자가 선택한 과목에 대하여 면접 질문을 생성하고 사용자의 대답을 평가합니다.
        참고: 답변의 최대 길이는 4097토큰입니다!
        - 사용자 대답을 튜터링 및 피드백을 합니다.
        - 새 세션을 시작하려면 페이지를 새로고침하면 됩니다.
        - 과목을 선택하고 시작해 즐겨보세요!
        """)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 면접 질문 안내 메시지
st.write("🎯 면접 질문")
if not st.session_state["messages"]:
    st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
else:
    st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")

# 새로운 질문 생성 버튼
if st.button("새로운 질문 생성", key="new_question_button"):
    try:
        # 예제 카테고리 및 매핑
        category = "example_category"
        directory_mapping = {"example_category": "./data/example"}
        PDF_FILE_MAPPING = {"example_category": "example.pdf"}
        
        # PDF 경로 생성
        current_dir = directory_mapping[category]
        pdf_filename = PDF_FILE_MAPPING[category]
        pdf_path = os.path.join(current_dir, pdf_filename)

        # PDF 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            st.error(f"PDF 파일이 없습니다: {pdf_path}")
            st.info("해당 카테고리의 PDF 파일 확인")
            st.stop()

        # PDF 처리 코드 (필요에 따라 구현)
        st.success("PDF 파일이 성공적으로 로드되었습니다!")

    except KeyError as e:
        # 매핑된 키가 없을 때
        st.error(f"카테고리 또는 매핑된 값이 잘못되었습니다: {e}")
    except Exception as e:
        # 기타 예외 처리
        st.error(f"오류가 발생했습니다: {e}")

# 채팅 입력 창
user_message = st.text_input("메시지를 입력하세요:", placeholder="여기에 메시지를 입력하세요...")

# 전송 버튼
if st.button("전송"):
    if user_message:
        st.session_state["chat_history"].append(f"사용자: {user_message}")

# 채팅 기록 표시
st.write("### 채팅 기록")
for message in st.session_state["chat_history"]:
    st.write(message)