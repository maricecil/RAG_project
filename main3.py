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

# API Key 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
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

# PDF 디렉토리에서 다수의 파일을 로드하는 함수
@st.cache_resource(show_spinner="면접질문을 준비하는 중입니다...")
def embed_files_from_directory(directory_path):
    try:
        # 디렉토리 경로 확인
        if not os.path.exists(directory_path):
            st.error(f"디렉토리가 존재하지 않습니다: {directory_path}")
            return None

        # 디렉토리에 PDF 파일이 있는지 확인
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        if not pdf_files:
            st.error(f"디렉토리에 PDF 파일이 없습니다: {directory_path}")
            return None

        # PyPDFDirectoryLoader를 사용하여 다수의 PDF 파일을 로드
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()

        if not docs:
            st.error("PDF 파일을 로드할 수 없습니다. 디렉토리를 확인하세요.")
            return None

        # 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)

        if not split_documents:
            st.error("문서를 분할할 수 없습니다. PDF 파일의 내용을 확인하세요.")
            return None

        # 임베딩(Embedding) 생성
        embeddings = OpenAIEmbeddings()

        # 벡터스토어 생성(Create DB) 및 저장
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

        # 검색기(Retriever) 생성
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"PDF 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return None

# 카테고리 선택
category = st.selectbox("카테고리를 선택하세요:", ["머신러닝", "네트워크", "통계", "파이썬"])

# 카테고리에 따라 디렉토리 설정
directory_mapping = {
    "머신러닝": "data/machine_learning/",
    "네트워크": "data/network/",
    "통계": "data/statistics/",
    "파이썬": "data/python/"
}

# 디렉토리 존재 여부 확인 및 생성
for path in directory_mapping.values():
    if not os.path.exists(path):
        os.makedirs(path)

# 카테고리에 따라 디렉토리 설정 및 체인 생성
if st.session_state["chain"] is None or st.session_state["selected_category"] != category:
    st.session_state["chain"] = embed_files_from_directory(directory_mapping[category])
    st.session_state["selected_category"] = category

# 프롬프트 로드 (백엔드에서만 처리)
question_prompt = load_prompt("prompts/question_prompt.yaml")  # 질문 생성용 프롬프트
feedback_prompt = load_prompt("prompts/feedback_prompt.yaml")  # 피드백 생성용 프롬프트

# 면접 질문 생성 및 표시
if st.session_state["chain"]:
    col1, col2 = st.columns([3, 1])
    with col1:
        # 기존 대화 내용 표시
        for i, message in enumerate(st.session_state["messages"]):
            st.write(f"Q: {message['question']}")
            # 답변이 있으면 표시 및 피드백 제공
            if message['answer']:
                st.write(f"A: {message['answer']}")
                
                # 답변에 대한 피드백이 없는 경우에만 생성
                if 'feedback' not in message:
                    retriever = st.session_state["chain"]
                    docs = retriever.get_relevant_documents(message['question'])
                    
                    if docs:
                        context = docs[0].page_content
                        feedback_chain = feedback_prompt | ChatOpenAI(temperature=0.3) | StrOutputParser()
                        feedback = feedback_chain.invoke({
                            "context": context,
                            "question": message['question'],
                            "answer": message['answer']
                        })
                        
                        st.session_state["messages"][i]["feedback"] = feedback
                        st.rerun()
                
                # 피드백 표시
                if 'feedback' in message:
                    st.write("💡 피드백:")
                    st.write(message['feedback'])
            else:
                # 사용자 답변 입력창
                user_answer = st.text_area("답변을 입력하세요:", key=f"answer_{i}")
                if st.button("답변 제출", key=f"submit_{i}"):
                    st.session_state["messages"][i]["answer"] = user_answer
                    st.rerun()
    
    # 새로운 질문 생성 버튼
    with col2:
        if st.button("새로운 질문 생성"):
            retriever = st.session_state["chain"]
            docs = retriever.get_relevant_documents(category)
            
            if docs:
                context = docs[0].page_content
                question_chain = question_prompt | ChatOpenAI(temperature=0.7) | StrOutputParser()
                question = question_chain.invoke({"context": context})
                
                st.session_state["messages"].append({"question": question, "answer": ""})
                st.rerun()