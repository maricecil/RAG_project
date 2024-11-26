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

# 메인 화면 구성: 왼쪽에는 대화 내용, 오른쪽에는 새 질문 버튼
if st.session_state["chain"]:
    col1, col2 = st.columns([3, 1])
    with col1:
        # 이전 대화 내용을 보여줍니다
        for i, message in enumerate(st.session_state["messages"]):
            st.write(f"{message['question']}")  # 질문 표시
            if message['answer']:  # 답변이 있다면
                st.write(f"A: {message['answer']}")  # 답변 표시
                
                # 답변에 대한 피드백이 아직 없다면
                if 'feedback' not in message:
                    retriever = st.session_state["chain"]
                    docs = retriever.get_relevant_documents(message['question'])
                    
                    # AI가 답변을 평가하고 피드백을 생성합니다
                    if docs:
                        context = docs[0].page_content
                        feedback_chain = feedback_prompt | ChatOpenAI(temperature=0.2) | StrOutputParser()
                        feedback = feedback_chain.invoke({
                            "context": context,
                            "question": message['question'],
                            "answer": message['answer']
                        })
                        
                        # 생성된 피드백을 저장하고 화면을 새로고침합니다
                        st.session_state["messages"][i]["feedback"] = feedback
                        st.rerun()
                
                # 피드백이 있다면 보여줍니다
                if 'feedback' in message:
                    st.write("💡 피드백:")
                    # 기본 피드백 표시
                    st.write(message['feedback'])
                    
                    # 예시 보기 버튼 추가
                    if st.button("💻 예시 보기", key=f"example_{i}"):
                        st.code("""
string = 'test'
capitalized_string = string.capitalize()
print(capitalized_string)
                        """)
            else:
                # 답변이 없다면 답변을 입력받을 텍스트 상자를 보여줍니다
                user_answer = st.text_area("답변을 입력하세요:", key=f"answer_{i}")
                if st.button("답변 제출", key=f"submit_{i}"):
                    st.session_state["messages"][i]["answer"] = user_answer
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
            st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
        else:  # 대화가 있다면
            st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")
            
        # 새로운 질문 생성 버튼
        if st.button("새로운 질문 생성"):
            current_dir = directory_mapping[category]
            pdf_filename = PDF_FILE_MAPPING[category]
            pdf_path = os.path.join(current_dir, pdf_filename)
            
            # PDF 파일이 있는지 확인합니다
            if not os.path.exists(pdf_path):
                st.error(f"PDF 파일이 없습니다: {pdf_path}")
                st.info("해당 카테고리의 PDF 파일을 확인해주세요.")
                st.stop()
            
            try:
                # PDF 내용을 읽어옵니다
                content = get_pdf_content(pdf_path)
                
                if content:
                    questions = []
                    lines = content.split('\n')  # PDF 내용을 줄별로 나눕니다
                    
                    # 이전에 나왔던 질문들을 기억합니다 (최근 5개)
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