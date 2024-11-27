# 필요한 라이브러리들을 가져옵니다
import os  # 파일/폴더 관리를 위한 라이브러리
import streamlit as st  # 웹 애플리케이션을 만드는 라이브러리
from dotenv import load_dotenv  # 환경변수를 불러오는 라이브러리
from langchain_core.messages.chat import ChatMessage  # AI 채팅 메시지를 다루는 도구
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI의 AI 모델을 사용하기 위한 도구
from langchain_core.output_parsers import StrOutputParser  # AI 출력을 문자열로 변환하는 도구
from langchain_teddynote.prompts import load_prompt  # AI 지시사항을 불러오는 도구
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 텍스트를 나누는 도구
from langchain_community.vectorstores import FAISS  # 텍스트를 검색하기 쉽게 저장하는 도구
from langchain_core.runnables import RunnablePassthrough  # AI 처리 과정을 연결하는 도구
from langchain_teddynote import logging  # 작업 기록을 남기는 도구
from langchain_community.document_loaders import PyPDFDirectoryLoader  # PDF 파일을 읽는 도구
from langchain.prompts import PromptTemplate  # AI 지시사항 템플릿을 만드는 도구
import random  # 무작위 선택을 위한 라이브러리
import PyPDF2  # PDF 파일을 다루는 라이브러리
import io  # 입출력 작업을 위한 라이브러리
import re  # 텍스트 패턴을 찾는 라이브러리
import json
from datetime import datetime
import pathlib

# 각 면접 주제(카테고리)와 영어 이름을 연결하는 사전
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
    "파이썬": "python.pdf",
    "머신러닝": "machine_learning.pdf",
    "딥러닝": "deep_learning.pdf",
    "데이터구조": "data_structure.pdf",
    "운영체제": "operating_system.pdf",
    "네트워크": "network.pdf",
    "통계": "statistics.pdf",
    "알고리즘": "algorithm.pdf"
}

# AI에게 줄 지시사항(프롬프트)을 파일에서 불러옵니다
feedback_prompt = load_prompt("prompts/feedback_prompt.yaml")  # 피드백용 지시사항
question_prompt = load_prompt("prompts/question_prompt.yaml")  # 질문용 지시사항

# OpenAI API 키 등의 중요 정보를 환경변수에서 불러옵니다
load_dotenv()

# 프로젝트 이름을 기록합니다
logging.langsmith("[Project] PDF_RAG")

# 필요한 폴더들이 없다면 새로 만듭니다
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 웹페이지의 제목을 설정합니다
st.title("AI 웹 개발자 면접 튜터링🚀")

# 대화 내용을 저장할 변수들을 초기화합니다
if "messages" not in st.session_state:  # 이전 대화가 없다면
    st.session_state["messages"] = []   # 새로운 대화 목록을 만듭니다
if "chain" not in st.session_state:     # AI 처리 과정이 없다면
    st.session_state["chain"] = None    # 새로 만들 준비를 합니다
if "selected_category" not in st.session_state:  # 선택된 주제가 없다면
    st.session_state["selected_category"] = None # 새로 선택할 준비를 합니다
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True  # 로그인 상태 유지를 위한 플래그

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

# 사용자가 선택할 수 있는 면접 주제들을 보여줍니다
category = st.selectbox("카테고리를 선택하세요:", ["파이썬", "머신러닝", "딥러닝", "데이터구조", "운영체제", "네트워크", "통계", "알고리즘"])

# 각 주제별로 PDF 파일이 있는 폴더 위치를 지정합니다
directory_mapping = {
    "파이썬": "data/python/",
    "머신러닝": "data/machine_learning/",
    "딥러닝": "data/deep_learning/",
    "데이터구조": "data/data_structure/",
    "운영체제": "data/operating_system/",
    "네트워크": "data/network/",
    "통계": "data/statistics/",
    "알고리즘": "data/algorithm/"
}

# 필요한 폴더들이 없다면 새로 만듭니다
for path in directory_mapping.values():
    if not os.path.exists(path):
        os.makedirs(path)

# 주제가 바뀌면 새로운 PDF 파일을 읽어옵니다
if st.session_state["chain"] is None or st.session_state["selected_category"] != category:
    st.session_state["chain"] = embed_files_from_directory(directory_mapping[category])
    st.session_state["selected_category"] = category

# PDF 파일을 읽어서 텍스트로 변환하는 함수
@st.cache_data
def get_pdf_content(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"PDF 파일 읽기 오류: {str(e)}")
        return None

# 메화 내용 저장을 위한 새로운 함수들
def get_chat_directory():
    chat_dir = pathlib.Path("chat_history")
    chat_dir.mkdir(exist_ok=True)
    return chat_dir

def save_chat_history(user_id, messages):
    chat_dir = get_chat_directory()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 사용자별 디렉토리 생성
    user_dir = chat_dir / user_id
    user_dir.mkdir(exist_ok=True)
    
    filename = f"{today}.json"
    filepath = user_dir / filename
    
    try:
        # 기존 대화 내용 불러오기
        existing_data = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        
        # 중복 제거를 위해 각 메시지의 고유 식별자 생성
        seen = set()
        unique_data = []
        for item in existing_data + messages:
            # 피드백까지 포함한 고유 식별자 생성
            identifier = f"{item['question']}_{item.get('answer', '')}_{item.get('feedback', '')}"
            if identifier not in seen:
                seen.add(identifier)
                # 피드백이 있는 메시지 우선 사용
                if 'feedback' in item:
                    for i, existing_item in enumerate(unique_data):
                        if existing_item['question'] == item['question'] and existing_item['answer'] == item.get('answer', ''):
                            unique_data[i] = item
                            break
                    else:
                        unique_data.append(item)
                else:
                    # 피드백이 없는 새 메시지 추가
                    unique_data.append(item)
        
        # 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"대화 내용 저장 중 오류 발생: {str(e)}")

def load_chat_history(user_id, date=None):
    chat_dir = get_chat_directory()
    user_dir = chat_dir / user_id
    
    if not user_dir.exists():
        return []
    
    if date:
        # 특정 날짜의 대든 대화 내용 로드
        files = sorted(user_dir.glob(f"{date}_*.json"))
        all_messages = []
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_messages.extend(json.load(f))
        return all_messages
    else:
        # 모든 대화 내용 로드
        all_messages = []
        for filepath in sorted(user_dir.glob("*.json")):
            with open(filepath, 'r', encoding='utf-8') as f:
                all_messages.extend(json.load(f))
        return all_messages

# Streamlit 세션 상태 초기화 부분 수정
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
    # 로컬 스토리지나 쿠키에서 사용자 정보 확인
    if 'current_user' in st.session_state:
        st.session_state["user_id"] = st.session_state['current_user']

# 로그인 섹션 수정
if not st.session_state["user_id"]:
    tab1, tab2 = st.tabs(["로그인", "회원가입"])
    
    with tab1:
        login_id = st.text_input("사용자 ID:", key="login_id")
        login_pw = st.text_input("비밀번호:", type="password", key="login_pw")
        if st.button("로그인"):
            users_file = pathlib.Path("users.json")
            if users_file.exists():
                with open(users_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                if login_id in users and users[login_id] == login_pw:
                    st.session_state["user_id"] = login_id
                    # 로그인 상태 유지를 위해 current_user 저장
                    st.session_state['current_user'] = login_id
                    st.session_state["messages"] = []
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 일치하지 않습니다.")
            else:
                st.error("등록된 사용자가 없습니다. 회원가입을 해주세요.")
    
    with tab2:
        new_id = st.text_input("새로운 ID:", key="new_id")
        new_pw = st.text_input("새로운 비밀번호:", type="password", key="new_pw")
        new_pw_confirm = st.text_input("비밀번호 확인:", type="password", key="new_pw_confirm")
        if st.button("회원가입"):
            if not new_id or not new_pw:
                st.error("ID와 비밀번호를 모두 입력해주세요.")
            elif new_pw != new_pw_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                users_file = pathlib.Path("users.json")
                if users_file.exists():
                    with open(users_file, 'r', encoding='utf-8') as f:
                        users = json.load(f)
                else:
                    users = {}
                
                if new_id in users:
                    st.error("이미 존재하는 ID입니다.")
                else:
                    users[new_id] = new_pw
                    with open(users_file, 'w', encoding='utf-8') as f:
                        json.dump(users, f, ensure_ascii=False, indent=2)
                    st.success("회원가입이 완료되었습니다. 로그인해주세요.")
    
    st.stop()

# 메인 화면 구성 바로 전에 추가 (st.title 다음, category selectbox 전에 위치)
if st.session_state["user_id"]:
    with st.sidebar:
        # 로그아웃 버튼 추가
        if st.button("로그아웃"):
            st.session_state["user_id"] = None
            if 'current_user' in st.session_state:
                del st.session_state['current_user']
            st.session_state["messages"] = []
            st.rerun()
            
        # 새로운 질문 생성 버튼을 사이드바로 이동
        st.write("🎯 면접 질문")
        if not st.session_state["messages"]:
            st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
        else:
            st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")
            
        if st.button("새로운 질문 생성"):
            current_dir = directory_mapping[category]
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
            
        # 기존의 대화 내용 관리 섹션
        st.write("💬 대화 내용 관리")
        chat_dir = get_chat_directory() / st.session_state["user_id"]
        if chat_dir.exists():
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
                        st.rerun()

# 메인 화면 구성: 대화 내용만 표시
if st.session_state["chain"]:
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
                    
                    # 현재 메시지에 피드백과 context 추가
                    st.session_state["messages"][i]["feedback"] = feedback
                    st.session_state["messages"][i]["context"] = context
                    
                    # 전체 메시지 저장
                    save_chat_history(st.session_state["user_id"], st.session_state["messages"])
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
        else:
            user_answer = st.text_area("답변을 입력하세요:", key=f"answer_{i}")
            if st.button("답변 제출", key=f"submit_{i}"):
                st.session_state["messages"][i]["answer"] = user_answer
                save_chat_history(st.session_state["user_id"], [st.session_state["messages"][i]])
                st.rerun()
