import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.embeddings import OpenAIEmbeddings

# PDF 경로 매핑
pdf_file_path = {
    "파이썬": "C:/Users/USER/REG_project/RAG_project/data/python/python.pdf",
    "머신러닝": "C:/Users/USER/REG_project/RAG_project/data/machine_learning/machine_learning.pdf",
    "딥러닝": "C:/Users/USER/REG_project/RAG_project/data/deep_learning/deep_learning.pdf",
    "데이터구조": "C:/Users/USER/REG_project/RAG_project/data/data_structure/data_structure.pdf",
    "운영체제": "C:/Users/USER/REG_project/RAG_project/data/operating_system/operating_system.pdf",
    "네트워크": "C:/Users/USER/REG_project/RAG_project/data/network/network.pdf",
    "통계": "C:/Users/USER/REG_project/RAG_project/data/statistics/statistics.pdf",
    "알고리즘": "C:/Users/USER/REG_project/RAG_project/data/algorithm/algorithm.pdf",
}

# 질문 파일 경로 매핑
questions_folder = "C:/Users/USER/REG_project/RAG_project/김준기님/"
topic_file_mapping = {
    "파이썬": "python_module.txt",
    "머신러닝": "ML_module.txt",
    "딥러닝": "DL_module.txt",
    "데이터구조": "DS_module.txt",
    "운영체제": "OS_module.txt",
    "네트워크": "Network_module.txt",
    "통계": "statistic_module.txt",
    "알고리즘": "algorithm_module.txt",
}

# FAISS 인덱스 저장 경로
faiss_index_folder = "C:/Users/USER/REG_project/RAG_project/faiss_indexes"
os.makedirs(faiss_index_folder, exist_ok=True)

# OpenAI API 키 입력
st.sidebar.title("AI 면접 튜터링 서비스")
api_key = st.sidebar.text_input("OpenAI API 키 입력", type="password")

if not api_key:
    st.sidebar.error("유효한 OpenAI API 키를 입력하세요!")
    st.stop()

# 주제 선택
selected_topic = st.sidebar.selectbox("주제를 선택하세요", list(pdf_file_path.keys()))

# PDF 처리 함수
def process_pdf(file_path, api_key):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# 질문 파일 로드 함수
def load_questions(topic):
    file_name = topic_file_mapping.get(topic)
    if not file_name:
        return []
    questions_file = os.path.join(questions_folder, file_name)
    if not os.path.exists(questions_file):
        return []
    with open(questions_file, "r", encoding="utf-8") as file:
        return [q.strip() for q in file.readlines() if q.strip()]

# FAISS 저장 및 로드 함수
def save_faiss_index(vectorstore, topic):
    sanitized_topic = topic_file_mapping.get(topic, topic).replace("_module.txt", "")
    index_path = os.path.join(faiss_index_folder, f"{sanitized_topic}_index")
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)

def load_faiss_index(topic, api_key):
    sanitized_topic = topic_file_mapping.get(topic, topic).replace("_module.txt", "")
    index_path = os.path.join(faiss_index_folder, f"{sanitized_topic}_index")
    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

# 질문 로드
questions = load_questions(selected_topic)

# FAISS 로드 및 생성
vectorstore = load_faiss_index(selected_topic, api_key)
if not vectorstore:
    pdf_path = pdf_file_path.get(selected_topic)
    if pdf_path and os.path.exists(pdf_path):
        texts = process_pdf(pdf_path, api_key)
        vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=api_key))
        save_faiss_index(vectorstore, selected_topic)
    else:
        st.error(f"{selected_topic} 주제의 PDF 파일을 찾을 수 없습니다.")

# 질문 선택
selected_question = st.sidebar.selectbox("질문 선택", questions)

# 메인 페이지 구성
st.title("AI 면접 튜터링")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if selected_question:
    st.session_state.chat_history.append(("질문", selected_question))
    st.chat_message("assistant").markdown(f"**{selected_question}**")

# 사용자 입력 처리
user_input = st.chat_input("질문이나 답변을 입력하세요...", key="user_chat_input")

if user_input:
    st.session_state.chat_history.append(("사용자", user_input))
    st.chat_message("user").markdown(user_input)

try:
    # ChatOpenAI 객체 초기화
    chat_model = ChatOpenAI(
        model="gpt-4",
        openai_api_key=api_key,
        temperature=0.7
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False,
    )

    # 사용자 응답에 대한 튜터링 및 피드백 생성
    prompt = f"""
    질문: {selected_question}
    사용자 응답: {user_input}
    
    아래의 내용을 작성하세요:
    1. 사용자의 응답에 대한 구체적인 피드백 (정확성, 관련성, 불완전성 등).
    2. 질문의 주제에 대한 간단한 추가 설명 (튜터링).
    3. 더 나은 답변을 작성할 수 있는 방향 제안.
    """
    response = chain({"question": prompt, "chat_history": st.session_state.chat_history})
    ai_response = response["answer"]

except Exception as e:
    ai_response = f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}"

# AI 응답 표시
st.session_state.chat_history.append(("튜터링", ai_response))
st.chat_message("assistant").markdown(ai_response)
