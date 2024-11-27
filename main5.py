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
import json  # JSON 파일을 다루는 라이브러리
from datetime import datetime  # 날짜와 시간을 다루는 라이브러리
import glob  # 파일 목록을 가져오는 라이브러리

# 파일 상단에 추가
PDF_FILE_MAPPING = {
    "파이썬": "python_questions.pdf",
    "머신러닝": "ml_questions.pdf",
    "딥러닝": "dl_questions.pdf",
    "데이터구조": "ds_questions.pdf",
    "운영체제": "os_questions.pdf",
    "네트워크": "network_questions.pdf",
    "통계": "statistics_questions.pdf",
    "알고리즘": "algorithm_questions.pdf"
}

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

# 각 카제별로 PDF 파일이 있는 폴더 위치를 지정합니다
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

# 이미 사용한 질문들을 기억하기 위한 변수를 초기화합니다
if "used_topics" not in st.session_state:
    st.session_state["used_topics"] = set()

# ChatSession 클래스 정의 추가
class ChatSession:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages = []
        self.category = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_session(self):
        if not os.path.exists("chat_sessions"):
            os.makedirs("chat_sessions")
        
        session_data = {
            "session_id": self.session_id,
            "messages": self.messages,
            "category": self.category,
            "timestamp": self.timestamp
        }
        
        with open(f"chat_sessions/session_{self.session_id}.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_session(session_id):
        try:
            with open(f"chat_sessions/session_{session_id}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                session = ChatSession()
                session.session_id = data["session_id"]
                session.messages = data["messages"]
                session.category = data["category"]
                session.timestamp = data["timestamp"]
                return session
        except Exception as e:
            st.error(f"세션 로드 중 오류 발생: {str(e)}")
            return None


# 세션 상태 초기화
if "current_session" not in st.session_state:
    st.session_state.current_session = ChatSession()

# 사이드바에 세션 관리 추가
with st.sidebar:
    st.subheader("대화 세션 관리")
    
    # 새 세션 시작 버튼
    if st.button("새 대화 시작"):
        st.session_state.current_session = ChatSession()
        st.session_state.messages = []
        st.rerun()
    
    # 구분선 추가
    st.divider()
    
    # 저장된 세션 목록
    session_files = glob.glob("chat_sessions/session_*.json")
    if session_files:
        st.subheader("저장된 대화 목록")
        sessions = []
        
        # 세션 파일들을 시간 순으로 정렬
        for file in sorted(session_files, reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 카테고리가 None인 경우 처리
                    category = data.get("category", "미분류")
                    sessions.append({
                        "id": data["session_id"],
                        "category": category if category else "미분류",
                        "timestamp": data["timestamp"]
                    })
            except Exception as e:
                st.error(f"세션 파일 읽기 오류: {file}")
                continue
        
        if sessions:  # 정상적으로 읽은 세션이 있는 경우
            # 세션 선택 드롭다운
            session_display = [f"{s['timestamp']} - {s['category']}" for s in sessions]
            selected_index = st.selectbox(
                "이전 대화 선택",
                range(len(session_display)),
                format_func=lambda i: session_display[i]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("불러오기", key="load_session"):
                    try:
                        session_id = sessions[selected_index]["id"]
                        loaded_session = ChatSession.load_session(session_id)
                        if loaded_session and loaded_session.category:
                            st.session_state.current_session = loaded_session
                            st.session_state.messages = loaded_session.messages
                            st.session_state.selected_category = loaded_session.category
                            st.success("이전 대화를 불러왔습니다.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"세션 불러오기 오류: {str(e)}")
            
            with col2:
                if st.button("삭제", key="delete_session"):
                    session_id = sessions[selected_index]["id"]
                    file_to_delete = f"chat_sessions/session_{session_id}.json"
                    try:
                        os.remove(file_to_delete)
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"세션 삭제 중 오류 발생: {e}")
    else:
        st.info("저장된 대화가 없습니다.")

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

# 메인 화면 구성: 왼쪽에는 대화 내용, 오른쪽에는 새 질문 버튼
if st.session_state["chain"]:
    col1, col2 = st.columns([3, 1])
    with col1:
        # 이전 대화 내용을 보여줍니다
        for i, message in enumerate(st.session_state["messages"]):
            st.write(f"Q: {message['question']}")  # 질문 표시
            if message['answer']:  # 답변이 있다면
                st.write(f"A: {message['answer']}")  # 답변 표시
                
                # 피드백 처리 (한 번만 실행되도록 수정)
                if 'feedback' not in message or not message['feedback']:
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
                        
                        # 피드백과 함께 context도 저장
                        st.session_state["messages"][i]["feedback"] = feedback
                        st.session_state["messages"][i]["context"] = context
                
                # 저장된 피드백 표시
                if 'feedback' in message and message['feedback']:
                    st.write("💡 피드백:")
                    st.write(message['feedback'])
                    
                    content = message['context']
                    # 현재 질문의 섹션 찾기
                    current_q = message['question']
                    start_idx = content.find(current_q)
                    
                    if start_idx != -1:
                        # 다음 질문 찾기 (백틱으로 시작하는 다음 질문)
                        next_q = content.find('`', start_idx + len(current_q) + 100)  # 여유 있게 검색
                        # 현재 질문의 섹션만 추출 (최소 100자 이상)
                        section = content[start_idx:next_q] if next_q != -1 else content[start_idx:]
                        

            else:
                # 답변이 없다면 답변을 입력받을 텍스트 상자를 보여줍니다
                user_answer = st.text_area("답변을 입력하세요:", key=f"answer_{i}")
                if st.button("답변 제출", key=f"submit_{i}"):
                    st.session_state["messages"][i]["answer"] = user_answer
                    # 세션 자동 저장
                    st.session_state.current_session.category = category
                    st.session_state.current_session.messages = st.session_state.messages
                    st.session_state.current_session.save_session()
                    st.rerun()
    
    with col2:
        # CSS로 버튼과 안내메시지가 스크롤을 따라가도록 설정합니다
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
        
        # 처음 시작하는 유저를 위해 상단에 안내 메시지를 추가합니다
        if not st.session_state["messages"]:  # 대화가 아직 없다면
            st.info("👇 아래 버튼을 눌러 첫 면접 문을 생성해보세요!")
        else:  # 대화가 있다면
            st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")
            
        # 새로 질문 생성 버튼
        if st.button("새로운 질문 생성"):
            current_dir = directory_mapping[category]
            
            # PDF 파일이 있는지 확인합니다
            if not os.path.exists(current_dir):
                st.error(f"디렉토리가 없습니다: {current_dir}")
                st.info("해당 카테고리의 디렉토리를 확인해주세요.")
                st.stop()
            
            pdf_files = [f for f in os.listdir(current_dir) if f.endswith('.pdf')]
            if not pdf_files:
                st.error(f"디렉토리에 PDF 파일이 없습니다: {current_dir}")
                st.info("해당 카테고리의 PDF 파일을 확인해주세요.")
                st.stop()
            
            try:
                # 첫 번째 PDF 파일을 사용
                pdf_path = os.path.join(current_dir, pdf_files[0])
                content = get_pdf_content(pdf_path)
                
                if content:
                    questions = []
                    lines = content.split('\n')  # PDF 내용을 줄별로 나눕니다
                    
                    # 이전에 나왔던 질문을 기억합니다 (최근 5개)
                    previous_questions = [msg["question"] for msg in st.session_state.get("messages", [])][-5:]
                    previous_questions_str = "\n".join(previous_questions) if previous_questions else "이전 질문 없음"
                    
                    # PDF의 각 줄을 확인하면서 질문을 찾습니다
                    for line in lines:
                        line = line.strip()
                        # 백틱(`)으로 둘러싸인 텍스트가 질문입니다
                        if (line.startswith('`') and line.endswith('`') and  
                            'instance' not in line.lower() and 
                            'error' not in line.lower() and 
                            'global var:' not in line.lower() and
                            len(line) > 10):
                            
                            # 백틱을 제거하고 질문 텍스트만 저장합니다
                            question_text = line.strip('`').strip()
                            questions.append(question_text)
                    
                    if questions:
                        # 이미 했던 질문은 제외합니다
                        used_questions = {msg["question"] for msg in st.session_state.get("messages", [])}
                        available_questions = [q for q in questions if q not in used_questions]
                        
                        # 새로운 질문이 있다면 랜덤으로 하나를 선택합니다
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


