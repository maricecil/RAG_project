import streamlit as st # 웹 애플리케이션을 만드는 라이브러리
import numpy as np
import random # 무작위 선택을 위한 라이브러리
import PyPDF2 # PDF 파일을 다루는 라이브러리
import io # 입출력 작업을 위한 라이브러리
import re # 텍스트 패턴을 찾는 라이브러리
import os # 파일/폴더 관리를 위한 라이브러리
import pathlib
# import getenv
import openai
from dotenv import load_dotenv # 환경변수를 불러오는 라이브러리
from langchain_core.messages.chat import ChatMessage # AI 채팅 메시지를 다루는 도구
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI의 AI 모델을 사용하기 위한 도구
from langchain_core.output_parsers import StrOutputParser # AI 출력을 문자열로 변환하는 도구
# from langchain_teddynote.prompts import load_prompt # AI 지시사항을 불러오는 도구
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 텍스트를 나누는 도구
from langchain_community.vectorstores import FAISS # 텍스트를 검색하기 쉽게 저장하는 도구
from langchain_core.runnables import RunnablePassthrough # AI 처리 과정을 연결하는 도구
# from langchain_teddynote import logging # 작업 기록을 남기는 도구
from langchain_community.document_loaders import PyPDFDirectoryLoader # PDF 파일을 읽는 도구
from langchain.prompts import PromptTemplate # AI 지시사항 템플릿을 만드는 도구
from streamlit_option_menu import option_menu
from PIL import Image
from utils import switch_page

# PDF 매핑 (먼저 정의해야 함)
CATEGORY_PDF_MAPPING = {
    "python": "C:/Users/USER/REG_project/RAG_project/data/python/python.pdf",
    "machine_learning": "C:/Users/USER/REG_project/RAG_project/data/machine_learning/machine_learning.pdf",
    "deep_learning": "C:/Users/USER/REG_project/RAG_project/data/deep_learning/deep_learning.pdf",
    "data_structure": "C:/Users/USER/REG_project/RAG_project/data/data_structure/data_structure.pdf",
    "operating_system": "C:/Users/USER/REG_project/RAG_project/data/operating_system/operating_system.pdf",
    "network": "C:/Users/USER/REG_project/RAG_project/data/network/network.pdf",
    "statistics": "C:/Users/USER/REG_project/RAG_project/data/statistics/statistics.pdf",
    "algorithm": "C:/Users/USER/REG_project/RAG_project/data/algorithm/algorithm.pdf",
}


# 세션 상태 초기화 함수
def initialize_session_state():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = None  # API 키 저장
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = list(CATEGORY_PDF_MAPPING.keys())[0]  # 기본 카테고리
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # 대화 기록 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # 질문/답변 초기화

# PDF 내용을 읽는 함수
def get_pdf_content(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            content = []
            for page in reader.pages:
                content.append(page.extract_text())
            return "\n".join(content)
    except Exception as e:
        st.error(f"PDF 내용 추출 중 오류 발생: {e}")
        return None

# PDF 임베딩 함수
@st.cache_resource(show_spinner="면접 질문을 준비하는 중입니다...")
def embed_pdf_file(pdf_path, openai_api_key):
    try:
        if not os.path.exists(pdf_path):
            st.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
            return None

        # PDF 내용 읽기
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            docs = [page.extract_text() for page in reader.pages if page.extract_text()]

        if not docs:
            st.error("PDF 파일이 비어 있습니다.")
            return None

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_text("\n".join(docs))

        # 텍스트 임베딩 생성
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(split_documents, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None




# 페이지 설정
st.set_page_config(page_title="무작위 질문")

# 세션 상태 초기화 호출
initialize_session_state()

# 사이드바: API 키 입력 및 섹션 선택
with st.sidebar:
    st.markdown("### AI 면접관")
    
    # API 키 입력
    openai_api_key = st.text_input("OpenAI API KEY", key="chatbot_api_key", type="password")
    if st.button("API 키 저장"):
        if openai_api_key.strip():
            st.session_state["api_key"] = openai_api_key
            st.success("API 키가 저장되었습니다!")
        else:
            st.error("유효한 API 키를 입력해주세요.")
    
    # 분야 선택
    selected_category = st.selectbox(
        "분야를 선택하세요:",
        list(CATEGORY_PDF_MAPPING.keys()),
        index=list(CATEGORY_PDF_MAPPING.keys()).index(st.session_state["selected_category"]),
    )
    st.session_state["selected_category"] = selected_category

    st.write("🎯 면접 질문")
    if not st.session_state["messages"]:
        st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
    else:
        st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")

# 새로운 질문 생성 버튼
    if st.button("새로운 질문 생성", key="new_question_btn"):
        # API 키 확인
        if "api_key" not in st.session_state or not st.session_state["api_key"].strip():
            st.error("API 키를 먼저 저장해주세요.")
            st.stop()
        category = st.session_state["selected_category"]
        pdf_path = CATEGORY_PDF_MAPPING[category]
        if not os.path.exists(pdf_path):
            st.error(f"PDF 파일이 없습니다: {pdf_path}")
            st.info("해당 카테고리의 PDF 파일을 확인해주세요.")
            st.stop()
        try:
            content = get_pdf_content(pdf_path)
            if content:
                questions = []
                lines = content.split("\n")

                # 최근 5개 질문 중복 방지
                previous_questions = [msg["question"] for msg in st.session_state.get("messages", [])][-5:]
                previous_questions_str = "\n".join(previous_questions) if previous_questions else "이전 질문 없음"

                for line in lines:
                    line = line.strip()
                    if line.startswith("`") and line.endswith("`") and len(line) > 10:
                        question_text = line.strip("`").strip()
                        questions.append(question_text)

                if questions:
                    used_questions = {msg["question"] for msg in st.session_state.get("messages", [])}
                    available_questions = [q for q in questions if q not in used_questions]

                    if available_questions:
                        question = random.choice(available_questions)
                        st.session_state["messages"].append({"question": question, "answer": ""})
                        st.session_state["current_question"] = question  # 현재 질문 저장
                    else:
                        st.warning("모든 질문을 완료했습니다. 다른 카테고리를 선택해주세요.")
                else:
                    st.warning("PDF에서 백틱(`)으로 둘러싸인 질문을 찾을 수 없습니다.")
            else:
                st.error("PDF에서 텍스트를 추출할 수 없습니다.")
        except Exception as e:
            st.error(f"PDF 파일 읽기 오류: {str(e)}")
            st.info("PDF 파일이 존재하고 읽기 가능한지 확인해주세요.")

# 메인 화면: ChatGPT 스타일 대화 인터페이스
st.markdown("## AI 면접관과 대화하기")
st.info("💬 이 세션에서는 AI 면접관과 채팅을 통해 질문과 답변을 주고받을 수 있습니다.")

# 이전 대화 내용 표시
if "chat_history" in st.session_state and st.session_state["chat_history"]:
    for chat in st.session_state["chat_history"]:
        if chat["role"] == "ai":
            st.markdown(f"**🤖 AI 면접관:** {chat['content']}")
        elif chat["role"] == "user":
            st.markdown(f"**👤 사용자:** {chat['content']}")

# 사용자 입력란
st.markdown("### 👇 질문에 대한 답변을 입력하세요:")
user_input = st.text_input("")

# 응답 처리
if st.button("전송"):
    if user_input.strip():
        # 사용자 입력 기록
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # AI 답변 생성 (예제, 실제 OpenAI API 호출 가능)
        ai_response = f"'{user_input}'에 대한 좋은 대답입니다! 추가로 다음 질문을 생각해보세요."
        st.session_state["chat_history"].append({"role": "ai", "content": ai_response})

        # 출력 업데이트
        st.experimental_rerun()
    else:
        st.warning("빈 입력은 허용되지 않습니다.")

# 질문 표시
if "current_question" in st.session_state and st.session_state["current_question"]:
    st.markdown(f"### 🧐 현재 질문: {st.session_state['current_question']}")
else:
    st.info("질문을 생성하려면 사이드바에서 선택하세요.")
