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

# 카테고리 매핑 추가
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

# PDF 파일명 매핑 추가
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

# 프롬프트 템플릿 로드
feedback_prompt = load_prompt("prompts/feedback_prompt.yaml")
question_prompt = load_prompt("prompts/question_prompt.yaml")

# API Key 정보로드
load_dotenv()

# 프로젝트 이름 설정
logging.langsmith("[Project] PDF_RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# Streamlit title
st.title("AI 웹 개발자 면접 튜터링🚀")

# 대화 초기화 및 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None
if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None

# 질문 주제 추적을 위한 세션 상태 추가
if "used_topics" not in st.session_state:
    st.session_state["used_topics"] = set()

# PDF 디렉토리에서 다수의 파일을 로드하는 함수
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
            st.error("PDF 파일을 로드할 수 없습니다. 디렉토리를 확인하세요.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)

        if not split_documents:
            st.error("문서를 분할할 수 없습니다. PDF 파일의 내용을 확인하세요.")
            return None

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"PDF 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return None

# 카테고리 선택
category = st.selectbox("카테고리를 선택하세요:", ["파이썬", "머신러닝", "딥러닝", "데이터구조", "운영체제", "네트워크", "통계", "알고리즘"])

# 카테고리에 따라 디렉토리 설정
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

# 디렉토리 존재 여부 확인 및 생성
for path in directory_mapping.values():
    if not os.path.exists(path):
        os.makedirs(path)

# 카테고리 변경 시 체인 재생성
if st.session_state["chain"] is None or st.session_state["selected_category"] != category:
    st.session_state["chain"] = embed_files_from_directory(directory_mapping[category])
    st.session_state["selected_category"] = category

# PDF 내용 캐싱 함수 추가
@st.cache_data
def get_pdf_content(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"PDF 파일 읽기 오류: {str(e)}")
        return None

# 면접 질문 생성 및 표시
if st.session_state["chain"]:
    col1, col2 = st.columns([3, 1])
    with col1:
        # 기존 대화 내용 표시
        for i, message in enumerate(st.session_state["messages"]):
            st.write(f"{message['question']}")
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
                        st.rerun()
                
                if 'feedback' in message:
                    st.write("💡 피드백:")
                    st.write(message['feedback'])
            else:
                user_answer = st.text_area("답변을 입력하세요:", key=f"answer_{i}")
                if st.button("답변 제출", key=f"submit_{i}"):
                    st.session_state["messages"][i]["answer"] = user_answer
                    st.rerun()
    
    with col2:
        if st.button("새로운 질문 생성"):
            current_dir = directory_mapping[category]
            pdf_filename = PDF_FILE_MAPPING[category]
            pdf_path = os.path.join(current_dir, pdf_filename)
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF 파일이 없습니다: {pdf_path}")
                st.info("해당 카테고리의 PDF 파일을 확인해주세요.")
                st.stop()
            
            try:
                # PDF 읽기 부분을 캐시된 함수로 대체
                content = get_pdf_content(pdf_path)
                
                if content:
                    questions = []
                    lines = content.split('\n')
                    
                    # 이전 질문들 가져오기 (최근 5개만)
                    previous_questions = [msg["question"] for msg in st.session_state.get("messages", [])][-5:]
                    previous_questions_str = "\n".join(previous_questions) if previous_questions else "이전 질문 없음"
                    
                    for line in lines:
                        line = line.strip()
                        if (line.startswith('`') and line.endswith('`') and  # 백틱으로 둘러싸인 라인 체크
                            'instance' not in line.lower() and 
                            'error' not in line.lower() and
                            'global var:' not in line.lower() and
                            len(line) > 10):
                            
                            # 질문 텍스트 정제 - 시작과 끝의 백틱 제거
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
