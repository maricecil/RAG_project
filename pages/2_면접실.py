# 1. 프로그램에 필요한 도구들을 가져옵니다
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
import json # JSON 형식의 데이터를 다루는 도구
from datetime import datetime # 날짜와 시간을 다루는 도구
import pathlib # 파일 경로를 더 쉽게 다루는 도구
import time
from utils import switch_page

# 2. 면접 주제와 관련 정보를 정리합니다
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

# 3. 각 주제별로 읽을 PDF 파일 이름을 정합니다
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

# 4. AI에게 줄 지시사항을 프롬프트에서 불러옵니다
feedback_prompt = load_prompt("C:/Users/user/Desktop/JSW/prompt/feedback_prompt.yaml")  # 피드백용 지시사항
question_prompt = load_prompt("C:/Users/user/Desktop/JSW/prompt/question_prompt.yaml")  # 질문용 지시사항

# 5. 프로그램 시작 전 기본 설정
load_dotenv() # OpenAI API 키 등의 중요 정보를 환경변수에서 불러옵니다
logging.langsmith("[Project] PDF_RAG") # 프로젝트 이름을 기록합니다

# 6. 필요한 폴더들을 만듭니다
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 7. 웹페이지의 제목을 설정합니다 # streamlit title
st.title("AI 개발자 면접 튜터링")

# 8. 대화 내용을 저장할 변수들을 초기화합니다
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None  # 초기값 설정
if "messages" not in st.session_state:  # 이전 대화가 없다면
    st.session_state["messages"] = []   # 새로운 대화 목록을 만듭니다
if "chain" not in st.session_state:     # AI 처리 과정이 없다면
    st.session_state["chain"] = None    # 새로 만들 준비를 합니다
if "selected_category" not in st.session_state:  # 선택된 주제가 없다면
    st.session_state["selected_category"] = None # 새로 선택할 준비를 합니다
if "authenticated" not in st.session_state: # 로그인 상태가 없다면
    st.session_state["authenticated"] = False # 로그인 준비를 합니다 # streamlit session_state
    
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
    
# 로그인된 사용자 확인
if not st.session_state["user_id"]:
    st.error("로그인되지 않았습니다. 먼저 로그인해주세요.")
    st.stop()
users = load_users()
current_user_id = st.session_state["user_id"]
if current_user_id not in users:
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
        환영합니다, {current_user_id}님!
    </div>
    """,
    unsafe_allow_html=True
)

# 이미 사용한 질문들을 기억하기 위한 변수를 초기화합니다
if "used_topics" not in st.session_state:
    st.session_state["used_topics"] = set() # 사용한 질문들을 저장할 집합

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

# 11. 사용자가 선택할 수 있는 면접 주제들을 보여줍니다 # streamlit selectbox
category = st.selectbox("카테고리를 선택하세요:", ["파이썬", "머신러닝", "딥러닝", "데이터구조", "운영체제", "네트워크", "통계", "알고리즘"])

# 12. 각 주제별로 PDF 파일이 있는 폴더 위치를 지정합니다
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

# 13. 필요한 폴더들이 없다면 새로 만듭니다
for path in directory_mapping.values():
    if not os.path.exists(path):
        os.makedirs(path)

# 14. 주제가 바뀌면 새로운 PDF 파일을 읽어옵니다
if st.session_state["chain"] is None or st.session_state["selected_category"] != category:
    st.session_state["chain"] = embed_files_from_directory(directory_mapping[category])
    st.session_state["selected_category"] = category

# 15. PDF 파일을 읽어서 텍스트로 변환하는 함수
@st.cache_data
def get_pdf_content(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"PDF 파일 읽기 오류: {str(e)}")
        return None
    
# 16. 대화 내용 저장을 위한 새로운 함수들
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
            json.dump(unique_data, f, ensure_ascii=False, indent=2) # json dump : 사용자 데이터 및 대화 내용 저장
            
    except Exception as e:
        st.error(f"대화 내용 저장 중 오류 발생: {str(e)}") # streamlit error

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
            with open(filepath, 'r', encoding='utf-8') as f: # utf-8 encoding : 문자열 처리
                all_messages.extend(json.load(f)) # json load : 사용자 데이터 및 대화 내용 불러오기
        return all_messages


    
# 새로운 질문 생성 버튼을 사이드바로 이동
st.sidebar.title("🎯 면접 질문")
if not st.session_state["messages"]:
    st.sidebar.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
else:
    st.sidebar.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")

if st.sidebar.button("새로운 질문 생성", key="new_question_btn"):
    # 질문 생성 관련 로직을 호출
    current_dir = directory_mapping[category]
    pdf_filename = PDF_FILE_MAPPING[category]
    pdf_path = os.path.join(current_dir, pdf_filename)
    
    if not os.path.exists(pdf_path):
        st.sidebar.error(f"PDF 파일이 없습니다: {pdf_path}")
        st.sidebar.info("해당 카테고리의 PDF 파일을 확인해주세요.")
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
                    st.rerun() # streamlit rerun
                else:
                    st.warning("모든 질문을 완료했습니다. 다른 카테고리를 선택해주세요.")
            else:
                st.warning("PDF에서 백틱(`)으로 둘러싸인 질문을 찾을 수 없습니다.") # streamlit warning
        else:
            st.error("PDF에서 텍스트를 추출할 수 없습니다.") # streamlit error
        
    except Exception as e:
        st.error(f"PDF 파일 읽기 오류: {str(e)}")
        st.info("PDF 파일이 존재하고 읽기 가능한지 확인해주세요.")
        
icon_path = "로고1.png"  # 아이콘 이미지 파일 경로
if os.path.exists(icon_path):  # 파일 존재 여부 확인
    st.sidebar.image(
        icon_path,
        use_container_width=True)  # 사이드바의 너비에 맞추기
else:
    st.sidebar.warning("아이콘 파일을 찾을 수 없습니다.")  # 경고 메시지 출력
    
    st.sidebar.markdown(f"**사용자: {st.session_state['user_id']}**")
if st.sidebar.button("로그아웃", key="logout_btn"):
    st.session_state["authenticated"] = False
    st.session_state["user_id"] = None
    st.rerun()
    
# 기존의 대화 내용 관리 섹션 # streamlit write
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

# 19. 메인 화면 구성: 대화 내용을 보여주고 관리하는 부분
if st.session_state["chain"]:  # AI 처리 과정이 준비되었다면
    # 지금까지 진행된 모든 대화를 하나씩 보여줍니다
    for i, message in enumerate(st.session_state["messages"]):
        # 구분선 추가 (첫 질문 제외)
        if i > 0:
            st.markdown("---")
            
        # 질문을 눈에 띄게 표시합니다
        st.markdown(f"### Q: {message['question']}")
        
        # 답변이 있는 경우의 처리
        if message['answer']:
            # 답변을 화시합니다
            st.markdown("#### A:")
            st.write(message['answer'])
            
            
            # 아직 피드백이 없는 경우 새로운 피드백을 생성합니다
            if 'feedback' not in message:
                retriever = st.session_state["chain"]
                docs = retriever.get_relevant_documents(message['question'])
                
                if docs:  # 관련 내용을 찾았다면
                    context = docs[0].page_content
                    feedback_chain = feedback_prompt | ChatOpenAI(
                        model="gpt-3.5-turbo-0125",
                        temperature=0.2
                    ) | StrOutputParser()
                    
                    feedback = feedback_chain.invoke({
                        "context": context,
                        "question": message['question'],
                        "answer": message['answer']
                    })
                    
                    # 생성된 피드백을 저장합니다
                    st.session_state["messages"][i]["feedback"] = feedback
                    st.session_state["messages"][i]["context"] = context
                    
                    # 대화 내용을 파일에 저장합니다
                    save_chat_history(st.session_state["user_id"], st.session_state["messages"])
                    st.rerun()
            
            # 피드백이 있는 경우 강조해서 표시합니다
            if 'feedback' in message:
                st.markdown("#### 💡 피드백:")
                feedback_text = message['feedback']
                
                # "핵심 키워드:" 부분을 굵은 글씨로 강조
                if "핵심 키워드:" in feedback_text:
                    feedback_text = feedback_text.replace("핵심 키워드:", "**핵심 키워드:**")
                
                # 각 항목 제목을 굵은 글씨로 강조
                items_to_bold = [
                    "1. 답변의 좋은 점",
                    "2. 보완이 필요한 점",
                    "3. 추가로 언급하면 좋을 내용",
                    "4. 개선된 답변 예시"
                ]
                
                for item in items_to_bold:
                    feedback_text = feedback_text.replace(item, f"**{item}**")
                
                st.markdown(feedback_text)
                
                # PDF 내용에서 현재 질문의 위치를 찾습니다
                content = message['context']
                current_q = message['question']
                start_idx = content.find(current_q)
                
                # 관련 내용을 추출합니다
                if start_idx != -1:
                    next_q = content.find('`', start_idx + len(current_q) + 100)
                    section = content[start_idx:next_q] if next_q != -1 else content[start_idx:]
        
        # 답변이 없는 경우 (새로운 질문인 경우)
        else:
            # 사용자가 답변을 입력할 수 있는 텍스트 영역을 표시합니다
            user_answer = st.text_area("답변을 입력하세요:", key=f"answer_{i}")
            # 답변 제출 버튼을 표시합니다
            if st.button("답변 제출", key=f"submit_{i}"):
                # 입력된 답변을 저장합니다
                st.session_state["messages"][i]["answer"] = user_answer
                # 답변을 파일에 저장합니다
                save_chat_history(st.session_state["user_id"], [st.session_state["messages"][i]])
                # 화면을 새로고침하여 피드백을 생성합니다
                st.rerun()