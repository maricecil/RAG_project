import streamlit as st # 웹 애플리케이션을 만드는 라이브러리
import numpy as np
import random # 무작위 선택을 위한 라이브러리
import PyPDF2 # PDF 파일을 다루는 라이브러리
import io # 입출력 작업을 위한 라이브러리
import re # 텍스트 패턴을 찾는 라이브러리
import os # 파일/폴더 관리를 위한 라이브러리
import pathlib
import getenv
import openai
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

# 각 카테고리별로 어떤 PDF 파일을 읽을지 정하는 사전
CATEGORY_PDF_MAPPING = {
    "python": "C:/Users/user/RAG_project/JSW/QnA/python.pdf",
    "machine_learning": "C:/Users/user/RAG_project/JSW/QnA/machine_learning.pdf",
    "deep_learning": "C:/Users/user/RAG_project/JSW/QnA/deep_learning.pdf",
    "data_structure": "C:/Users/user/RAG_project/JSW/QnA/data_structure.pdf",
    "operating_system": "C:/Users/user/RAG_project/JSW/QnA/operating_system.pdf",
    "network": "C:/Users/user/RAG_project/JSW/QnA/network.pdf",
    "statistics": "C:/Users/user/RAG_project/JSW/QnA/statistics.pdf",
    "algorithm": "C:/Users/user/RAG_project/JSW/QnA/algorithm.pdf"
}

# 세션 상태 초기화
def initialize_session_state():
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = list(CATEGORY_PDF_MAPPING.keys())[0]  # 기본값
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "used_topics" not in st.session_state:
        st.session_state["used_topics"] = set()
        
initialize_session_state()

# PDF 임베딩 함수
@st.cache_resource(show_spinner="면접 질문을 준비하는 중입니다...")
def embed_pdf_file(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            st.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
            return None

        # PDF 읽기
        loader = PyPDF2.PdfReader(open(pdf_path, "rb"))
        docs = [page.extract_text() for page in loader.pages if page.extract_text()]
        
        if not docs:
            st.error("PDF 파일이 비어 있습니다. 내용을 확인하세요.")
            return None

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_text("\n".join(docs))

        # 텍스트 임베딩
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None

# 사이드바
with st.sidebar:
    
    st.markdown("### AI 면접관")
    selected_category = st.selectbox(
        "분야를 선택하세요:",
        list(CATEGORY_PDF_MAPPING.keys()),
        index=list(CATEGORY_PDF_MAPPING.keys()).index(st.session_state["selected_category"]),
    )
    st.session_state["selected_category"] = selected_category

# 메인 화면
st.markdown("## 무작위 질문 생성")
st.info(
    """
    📚 이 세션에서는 AI 면접관이 사용자가 선택한 과목에 대해 면접 질문을 생성하고 사용자의 대답을 평가합니다.
    참고: 답변의 최대 길이는 4097토큰입니다!
    """
)

# 질문 생성 버튼
if st.button("새로운 질문 생성", key="new_question_button"):
    try:
        category = st.session_state["selected_category"]
        pdf_path = CATEGORY_PDF_MAPPING[category]
        
        if not os.path.exists(pdf_path):
            st.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            st.stop()

        retriever = embed_pdf_file(pdf_path)
        if retriever:
            st.success(f"{category} PDF 파일이 성공적으로 처리되었습니다.")
    except Exception as e:
        st.error(f"질문 생성 중 오류 발생: {e}")

# 사용자 입력
st.markdown("### 면접 질문에 대한 응답을 입력하세요:")
user_message = st.text_input("메시지를 입력하세요:", placeholder="여기에 메시지를 입력하세요...")

# 전송 버튼
if st.button("전송"):
    if user_message.strip():  # 빈 메시지 방지
        st.session_state["chat_history"].append(f"사용자: {user_message}")
    else:
        st.warning("빈 메시지는 입력할 수 없습니다.")

# 채팅 기록 표시
st.markdown("### 채팅 기록")
if st.session_state["chat_history"]:
    for message in st.session_state["chat_history"]:
        st.write(message)