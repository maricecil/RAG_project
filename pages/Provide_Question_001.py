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
import yaml
from pathlib import Path
import json

# DebuggablePassThrough 정의
class DebuggablePassThrough:
    def __init__(self, name="Debug"):
        self.name = name

    def __call__(self, data):
        print(f"[{self.name}] Data: {data}")
        return data

# 프롬프트 파일 불러오기 함수
def load_prompt_with_debug(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)  # YAML 파일 로드
            # DebuggablePassThrough로 디버깅
            data = DebuggablePassThrough("YAML Loaded")(data)
        return data
    except Exception as e:
        raise RuntimeError(f"프롬프트 파일 로드 중 오류 발생: {e}")
    
# OpenAI API를 사용한 피드백 생성 함수
def generate_feedback(answer, prompt, api_key):
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant who provides feedback based on the following prompt: {prompt}"},
                {"role": "user", "content": f"User's answer: {answer}"}
            ]
        )
        feedback = response["choices"][0]["message"]["content"]
        return feedback
    except Exception as e:
        return f"피드백 생성 중 오류 발생: {e}"

# 페이지 설정
st.set_page_config(page_title="무작위 질문")

# PDF 매핑 (먼저 정의해야 함)
CATEGORY_PDF_MAPPING = {
    "선택하지 않음": "C:/Users/user/RAG_project/JSW/QnA/none.pdf",
    "python": "C:/Users/user/RAG_project/JSW/QnA/python.pdf",
    "machine_learning": "C:/Users/user/RAG_project/JSW/QnA/machine_learning.pdf",
    "deep_learning": "C:/Users/user/RAG_project/JSW/QnA/deep_learning.pdf",
    "data_structure": "C:/Users/user/RAG_project/JSW/QnA/data_structure.pdf",
    "operating_system": "C:/Users/user/RAG_project/JSW/QnA/operating_system.pdf",
    "network": "C:/Users/user/RAG_project/JSW/QnA/network.pdf",
    "statistics": "C:/Users/user/RAG_project/JSW/QnA/statistics.pdf",
    "algorithm": "C:/Users/user/RAG_project/JSW/QnA/algorithm.pdf",
}

#사용자 데이터 경로부터 로그인된 사용자 확인까지 main_page.py에서 사용자가 입력한 로그인 정보를 불러오고 확인하는 코드입니다.

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

# 로그인된 사용자 확인
if not st.session_state["user_id"]:
    st.error("로그인되지 않았습니다. 먼저 로그인해주세요.")
    st.stop()

users = load_users()
current_user_id = st.session_state["user_id"]

if current_user_id not in users:
    st.error("사용자 정보를 확인할 수 없습니다. 다시 로그인해주세요.")
    st.stop()

st.write(f"환영합니다, {current_user_id}님!")

# 세션 상태 초기화 함수
def initialize_session_state():
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None  # 초기값 설정
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # 초기값 설정
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = list(CATEGORY_PDF_MAPPING.keys())[0]  # 기본 카테고리
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""  # API 키 초기화
        
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

# 프롬프트 파일 불러오기
def load_prompt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            prompt_data = yaml.safe_load(file)  # YAML 파일 읽기
        return prompt_data
    except Exception as e:
        raise RuntimeError(f"프롬프트 파일 로드 중 오류 발생: {e}")

feedback_prompt = load_prompt("C:/Users/user/RAG_project/JSW/prompt/feedback_prompt.yaml")  # 피드백용 지시사항

# 채팅 디렉토리를 반환하는 함수 정의
def get_chat_directory():
    # 현재 파일의 디렉토리 경로를 가져옵니다.
    base_dir = Path(__file__).parent
    # 'chat_history' 폴더를 기준으로 채팅 기록을 저장할 디렉토리를 설정합니다.
    chat_dir = base_dir / "chat_history"
    # 디렉토리가 존재하지 않으면 생성합니다.
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir


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
        index=list(CATEGORY_PDF_MAPPING.keys()).index("선택하지 않음"),
    )
    st.session_state["selected_category"] = selected_category

    # 선택하지 않음일 경우 안내 메시지
    if selected_category == "선택하지 않음":
        st.warning("PDF 파일을 선택하지 않았습니다. 질문 생성을 위해 카테고리를 선택해주세요.")
    else:
        st.info(f"선택된 카테고리: {selected_category}")

    st.write("🎯 면접 질문")
    if not st.session_state["messages"]:
        st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
    else:
        st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")

    # 새로운 질문 생성 버튼
    if st.button("새로운 질문 생성", key="new_question_btn"):

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

                # 질문 중복 방지
                previous_questions = [msg["question"] for msg in st.session_state.get("messages", [])]
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

# 기존의 대화 내용 관리 섹션
st.write("💬 대화 내용 관리")

# 사용자의 채팅 디렉토리를 설정합니다.
chat_dir = get_chat_directory() / st.session_state["user_id"]

# 사용자의 채팅 디렉토리가 없으면 생성합니다.
chat_dir.mkdir(parents=True, exist_ok=True)

if chat_dir.exists():
    # JSON 파일 목록을 가져와서 날짜 리스트를 생성합니다.
    available_dates = [f.stem for f in chat_dir.glob("*.json")]
    if available_dates:
        view_date = st.selectbox(
            "날짜 선택:",
            sorted(available_dates, reverse=True),
            key="view_date"
        )
        if st.button("선택한 날짜 보기"):
            # 선택한 날짜의 대화 내용을 불러와서 session_state에 저장
            chat_file = chat_dir / f"{view_date}.json"
            if chat_file.exists():
                with open(chat_file, 'r', encoding='utf-8') as f:
                    st.session_state["messages"] = json.load(f)
                st.experimental_rerun()
            else:
                st.error("선택한 날짜의 채팅 기록이 존재하지 않습니다.")
    else:
        st.info("저장된 대화 내용이 없습니다.")
else:
    st.info("저장된 대화 내용이 없습니다.")
    
# 프롬프트 파일 불러오기
prompt_path = "C:/Users/user/RAG_project/JSW/prompt/feedback_prompt.yaml"
try:
    feedback_prompt = load_prompt_with_debug(prompt_path)
    st.success("프롬프트 파일 로드 완료!")
    st.write(feedback_prompt)  # Streamlit에 출력
except Exception as e:
    st.error(f"프롬프트 파일 로드 실패: {e}")

# 중앙 화면에 질문 출력
st.markdown("## 무작위 질문 생성")

if "messages" in st.session_state and st.session_state["messages"]:
    for idx, msg in enumerate(st.session_state["messages"]):
        with st.container():
            # 질문 출력
            st.markdown(f"### **❓ 질문 {idx+1}:** {msg['question']}")
            
            # 응답 출력
            if msg.get("answer"):
                st.markdown(f"**💬 응답:** {msg['answer']}")
            else:
                st.markdown("💬 아직 응답이 작성되지 않았습니다.")

            # 피드백 생성 및 출력
            if msg.get("answer") and "api_key" in st.session_state and st.session_state["api_key"]:
                feedback = generate_feedback(
                    msg["answer"],
                    feedback_prompt["feedback_prompt"]["instructions"],  # YAML에서 피드백 지침 사용
                    st.session_state["api_key"]
                )
                st.markdown(f"**📝 피드백:** {feedback}")
            else:
                st.info("피드백을 생성하려면 OpenAI API 키를 입력하세요.")
            st.markdown("---")
else:
    st.info("생성된 질문이 없습니다. 새로운 질문을 생성하세요.")

# 사용자 입력
st.markdown("### 면접 질문에 대한 응답을 입력하세요:")
user_message = st.text_input("메시지를 입력하세요:", placeholder="여기에 메시지를 입력하세요...")

# 전송 버튼
if st.button("응답 저장"):
    if user_message.strip():
        if "messages" in st.session_state and st.session_state["messages"]:
            st.session_state["messages"][-1]["answer"] = user_message
            st.session_state["chat_history"].append(f"사용자: {user_message}")
            st.success("응답이 저장되었습니다.")
        else:
            st.warning("응답을 저장할 질문이 없습니다. 먼저 질문을 생성하세요.")
    else:
        st.warning("빈 메시지는 입력할 수 없습니다.")