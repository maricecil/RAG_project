import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
import random
import PyPDF2
import io
import re
import json
from datetime import datetime
import pathlib
import pandas as pd
import uuid
import time

# 상수 정의
CACHE_DIR = ".cache"
CHAT_DIR = "chat_history"
DATA_DIR = "data"

# 카테고리 매핑
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

# PDF 파일 매핑
PDF_FILE_MAPPING = {
    "파이썬": "python.pdf",
    "머신러닝": "machine_learning.pdf",
    "딥러닝": "deep_learning.pdf",
    "데이터구조": "data_structure.pdf",
    "운영체제": "operating_system.pdf",
    "네트워크": "network.pdf",
    "통계": "statistics.pdf",
    "알고리즘": "algorithm.pdf"
}

# 환경 변수 및 프롬프트 로드
load_dotenv()
feedback_prompt = load_prompt("prompts/feedback_prompt.yaml")
question_prompt = load_prompt("prompts/question_prompt.yaml")

# 세션 상태 초기화 함수
def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.update({
            "messages": [],
            "chain": None,
            "selected_category": None,
            "used_topics": set(),
            "user_id": None,
            "initialized": True
        })

# 환경 변수 검증
def validate_environment():
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.error(f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        return False
    return True

# 디렉토리 생성
def create_required_directories():
    directories = [
        CACHE_DIR,
        f"{CACHE_DIR}/files",
        f"{CACHE_DIR}/embeddings",
        CHAT_DIR,
        *[f"{DATA_DIR}/{category}" for category in CATEGORY_MAPPING.values()]
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# PDF 처리 함수
@st.cache_resource(show_spinner="면접질문을 준비하는 중입니다...")
def embed_files_from_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            st.error(f"디렉토리가 존재하지 않습니다: {directory_path}")
            return None

        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        if not pdf_files:
            st.error(f"디렉토리에 PDF 파일이 없습니다: {directory_path}")
            return None

        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()

        if not docs:
            st.error("PDF 파일을 로드할 수 없습니다.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {str(e)}")
        return None

@st.cache_data
def get_pdf_content(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"PDF 파일 읽기 오류: {str(e)}")
        return None

# 채팅 기록 관리 함수들
def get_chat_directory():
    chat_dir = pathlib.Path(CHAT_DIR)
    chat_dir.mkdir(exist_ok=True)
    return chat_dir

def save_chat_history(user_id, messages):
    try:
        chat_dir = get_chat_directory()
        today = datetime.now().strftime("%Y-%m-%d")
        user_dir = chat_dir / user_id
        user_dir.mkdir(exist_ok=True)
        filepath = user_dir / f"{today}.json"
        
        existing_data = []
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                st.error("채팅 기록 파일이 손상되었습니다.")
                return False
        
        existing_data.extend(messages)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        return True
            
    except Exception as e:
        st.error(f"채팅 기록 저장 중 오류 발생: {str(e)}")
        return False

def load_chat_history(user_id, date=None):
    chat_dir = get_chat_directory()
    user_dir = chat_dir / user_id
    
    if not user_dir.exists():
        return []
    
    if date:
        filepath = user_dir / f"{date}.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    else:
        all_messages = []
        for filepath in user_dir.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                all_messages.extend(json.load(f))
        return all_messages

def get_available_dates(user_id):
    chat_dir = pathlib.Path(CHAT_DIR) / user_id
    if not chat_dir.exists():
        return []
    
    dates = []
    for file in chat_dir.glob("*.json"):
        date = file.stem
        dates.append(date)
    
    return sorted(dates, reverse=True)

def delete_chat_history(user_id, date):
    chat_dir = pathlib.Path(CHAT_DIR) / user_id
    file_path = chat_dir / f"{date}.json"
    
    try:
        if file_path.exists():
            file_path.unlink()
            return True
    except Exception as e:
        st.error(f"채팅 기록 삭제 중 오류 발생: {str(e)}")
        return False

# 사용자 인증 관련 함수들
def load_user_credentials():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_user_credentials(credentials):
    with open('users.json', 'w') as f:
        json.dump(credentials, f)

def verify_credentials(user_id, password):
    try:
        credentials = load_user_credentials()
        if not credentials:
            st.error("사용자 데이터를 불러올 수 없습니다.")
            return False
            
        if user_id not in credentials:
            st.error("존재하지 않는 사용자입니다.")
            return False
            
        return credentials[user_id] == password
        
    except Exception as e:
        st.error(f"인증 중 오류 발생: {str(e)}")
        return False

def register_new_user(user_id, password, password_confirm):
    if not user_id or not password:
        st.error("아이디와 비밀번호를 모두 입력해주세요.")
        return False
    
    if password != password_confirm:
        st.error("비밀번호가 일치하지 않습니다.")
        return False
    
    credentials = load_user_credentials()
    if user_id in credentials:
        st.error("이미 존재하는 아이디입니다.")
        return False
    
    credentials[user_id] = password
    save_user_credentials(credentials)
    return True

# 채팅 기록 관리자 UI
def show_chat_history_manager():
    st.sidebar.title("대화 기록 관리")
    
    # 1. 현재 상태 확인을 위한 디버깅 출력 추가
    st.sidebar.write("현재 사용자:", st.session_state["user_id"])
    
    dates = get_available_dates(st.session_state["user_id"])
    if not dates:
        st.sidebar.write("저장된 대화 기록이 없습니다.")
        return
    
    # 2. 사용 가능한 날짜 확인을 위한 디버깅 출력
    st.sidebar.write("사용 가능한 날짜:", dates)
    
    unique_id = str(uuid.uuid4())
    selected_date = st.sidebar.selectbox(
        "날짜 선택",
        dates,
        key=f"date_selector_{unique_id}"
    )
    
    # 3. 선택된 날짜 확인을 위한 디버깩 출력
    st.sidebar.write("선택된 날짜:", selected_date)
    
    if st.sidebar.button("대화 내용 삭제", key=f"delete_button_{unique_id}"):
        chat_dir = pathlib.Path(CHAT_DIR) / st.session_state["user_id"]
        file_path = chat_dir / f"{selected_date}.json"
        
        # 4. 삭제 시도
        st.sidebar.write("삭제 시도")
        st.sidebar.write("삭제할 파일 경로:", file_path)
        st.sidebar.write("파일 존재 여부:", file_path.exists())
        
        try:
            if file_path.exists():
                file_path.unlink()
                st.session_state["messages"] = []
                st.sidebar.success("대화 내용이 삭제되었습니다!")
                st.rerun()
            else:
                st.sidebar.error("삭제할 파일을 찾을 수 없습니다.")
        except Exception as e:
            st.sidebar.error(f"삭제 중 오류 발생: {str(e)}")
            st.sidebar.write("상세 오류:", str(e))

# 메인 애플리케이션 UI
def main():
    st.title("AI 웹 개발자 면접 튜터링🚀")
    
    # 로그인 상태가 아닐 때
    if "user_id" not in st.session_state or not st.session_state["user_id"]:
        tab1, tab2 = st.tabs(["로그인", "회원가입"])
        
        with tab1:
            user_id = st.text_input("사용자 ID:", key="login_id")
            password = st.text_input("비밀번호:", type="password", key="login_pw")
            
            if st.button("로그인", key="login_button"):
                if verify_credentials(user_id, password):
                    st.session_state["user_id"] = user_id
                    today = datetime.now().strftime("%Y-%m-%d")
                    st.session_state["messages"] = load_chat_history(user_id, today)
                    st.success("로그인 성공!")
                    st.rerun()
        
        with tab2:
            new_id = st.text_input("새로운 사용자 ID:", key="register_id")
            new_pw = st.text_input("새로운 비밀번호:", type="password", key="register_pw1")
            new_pw_confirm = st.text_input("비밀번호 확인:", type="password", key="register_pw2")
            
            if st.button("회원가입", key="register_button"):
                if register_new_user(new_id, new_pw, new_pw_confirm):
                    st.success("회원가입이 완료되었습니다!")
        
        st.stop()
    
    # 로그인 상태일 때
    col1, col2 = st.columns([6,1])
    with col2:
        if st.button("로그아웃", type="secondary", key=f"logout_button_{uuid.uuid4()}"):
            st.session_state.clear()
            st.rerun()
    
    show_chat_history_manager()

    # 카테고리 선택 및 면접 질문 생성 UI
    category = st.selectbox("카테고리를 선택하세요:", list(CATEGORY_MAPPING.keys()))
    
    # 주제가 바뀌면 새로운 PDF 파일을 읽어옵니다
    if st.session_state["chain"] is None or st.session_state["selected_category"] != category:
        st.session_state["chain"] = embed_files_from_directory(f"{DATA_DIR}/{CATEGORY_MAPPING[category]}")
        st.session_state["selected_category"] = category

    # 메인 화면 구성: 왼쪽에는 대화 내용, 오른쪽에는 새 질문 버튼
    if st.session_state["chain"]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 이전 대화 내용을 보여줍니다
            for i, message in enumerate(st.session_state["messages"]):
                st.write(f"Q: {message['question']}")
                if message['answer']:
                    st.write(f"A: {message['answer']}")
                    
                    if 'feedback' not in message:
                        retriever = st.session_state["chain"]
                        docs = retriever.get_relevant_documents(message['question'])
                        
                        if docs:
                            context = docs[0].page_content
                            feedback_chain = feedback_prompt | ChatOpenAI(temperature=0.2) | StrOutputParser()
                            feedback = feedback_chain.invoke({
                                "context": context,
                                "question": message['question'],
                                "answer": message['answer']
                            })
                            
                            st.session_state["messages"][i]["feedback"] = feedback
                            st.session_state["messages"][i]["context"] = context
                            st.rerun()
                    
                    if 'feedback' in message:
                        st.write("💡 피드백:")
                        st.write(message['feedback'])
                        
                        content = message['context']
                        current_q = message['question']
                        start_idx = content.find(current_q)
                        
                        if start_idx != -1:
                            next_q = content.find('`', start_idx + len(current_q) + 100)
                            section = content[start_idx:next_q] if next_q != -1 else content[start_idx:]
                            st.write(section)
                else:
                    user_answer = st.text_area("답변을 입력하세요:", key=f"answer_input_{i}")
                    if st.button("답변 제출", key=f"submit_button_{i}"):
                        st.session_state["messages"][i]["answer"] = user_answer
                        save_chat_history(st.session_state["user_id"], [st.session_state["messages"][i]])
                        st.rerun()
        
        with col2:
            st.markdown(
                """
                <style>
                div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] {
                    position: sticky;
                    top: 3rem;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            if not st.session_state["messages"]:
                st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
            else:
                st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")

 # 새로운 질문 생성 버튼
            if st.button("새로운 질문 생성", key=f"new_question_{uuid.uuid4()}"):
                current_dir = f"{DATA_DIR}/{CATEGORY_MAPPING[category]}"
                pdf_filename = PDF_FILE_MAPPING[category]
                pdf_path = os.path.join(current_dir, pdf_filename)
                
                if not os.path.exists(pdf_path):
                    st.error(f"PDF 파일이 없습니다: {pdf_path}")
                    st.info("해당 카테고리의 PDF 파일을 확인해주세요.")
                    st.stop()
                
                try:
                    content = get_pdf_content(pdf_path)
                    
                    if content:
                        questions = []
                        lines = content.split('\n')
                        
                        previous_questions = [msg["question"] for msg in st.session_state.get("messages", [])][-5:]
                        previous_questions_str = "\n".join(previous_questions) if previous_questions else "이전 질문 없음"
                        
                        for line in lines:
                            line = line.strip()
                            if (line.startswith('`') and line.endswith('`') and  
                                'instance' not in line.lower() and 
                                'error' not in line.lower() and 
                                'global var:' not in line.lower() and
                                len(line) > 10):
                                
                                question_text = line.strip('`').strip()
                                questions.append(question_text)
                        
                        if questions:
                            used_questions = {msg["question"] for msg in st.session_state.get("messages", [])}
                            available_questions = [q for q in questions if q not in used_questions]
                            
                            if available_questions:
                                question = random.choice(available_questions)
                                st.session_state["messages"].append({"question": question, "answer": ""})
                                st.rerun()
                            else:
                                st.warning("모든 질문을 완료했습니다. 다른 카테고리를 선택해주세요.")
                        else:
                            st.warning("PDF에서 백틱(`)으로 둘러싸인 질문을 찾을 수 없습니다.")
                    else:
                        st.error("PDF에서 텍스트를 추출할 수 없습니다.")
                    
                except Exception as e:
                    st.error(f"PDF 파일 읽기 오류: {str(e)}")
                    st.info("PDF 파일이 존재하고 읽기 가능한지 확인해주세요.")

if __name__ == "__main__":
    # 환경 변수 검증
    if not validate_environment():
        st.stop()
    
    # 필요한 디렉토리 생성
    create_required_directories()
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 로깅 설정
    logging.langsmith("[Project] PDF_RAG")
    
    # 메인 앱 실행
    main()