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

# API 키 확인 및 유효성 검사
if not api_key:
    st.sidebar.error("유효한 OpenAI API 키를 입력하세요!")
    st.stop()

# 주제 선택
selected_topic = st.sidebar.selectbox("주제를 선택하세요", list(pdf_file_path.keys()))

# PDF 처리 함수
def process_pdf(file_path, api_key):
    """PDF 파일을 처리하여 텍스트 청크를 반환."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = text_splitter.split_documents(documents)
    return split_texts

# 질문 파일 로드 함수
def load_questions(topic):
    """카테고리별 질문 파일을 로드."""
    file_name = topic_file_mapping.get(topic)
    if not file_name:
        return []
    questions_file = os.path.join(questions_folder, file_name)
    if not os.path.exists(questions_file):
        return []
    with open(questions_file, "r", encoding="utf-8") as file:
        questions = file.readlines()
    return [q.strip() for q in questions if q.strip()]  # 공백 제거

# FAISS 저장 및 로드 함수
def save_faiss_index(vectorstore, topic):
    """FAISS 인덱스를 주제별로 저장."""
    sanitized_topic = topic_file_mapping.get(topic, topic).replace("_module.txt", "")
    index_path = os.path.join(faiss_index_folder, f"{sanitized_topic}_index")
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)

def load_faiss_index(topic, api_key):
    """주제별로 저장된 FAISS 인덱스를 로드."""
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
    st.sidebar.info(f"{selected_topic} 주제의 PDF를 처리 중입니다...")
    pdf_path = pdf_file_path.get(selected_topic)
    if pdf_path and os.path.exists(pdf_path):
        texts = process_pdf(pdf_path, api_key)
        vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=api_key))
        save_faiss_index(vectorstore, selected_topic)
        st.sidebar.success(f"{selected_topic} 주제의 FAISS 벡터스토어를 생성했습니다!")
    else:
        st.error(f"{selected_topic} 주제의 PDF 파일을 찾을 수 없습니다.")

# 질문 선택
selected_question = st.sidebar.selectbox("질문 선택", questions)

# 메인 페이지 구성
st.title("AI 면접 튜터링")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 선택된 질문을 채팅 기록에 추가
if selected_question:
    st.session_state.chat_history.append(("질문", selected_question))
    st.chat_message("assistant").markdown(f"**{selected_question}**")

# 사용자 입력 처리
user_input = st.chat_input("질문이나 답변을 입력하세요...", key="user_chat_input")

if user_input:
    # 사용자 입력 기록 추가
    st.session_state.chat_history.append(("사용자", user_input))
    st.chat_message("user").markdown(user_input)

try:
    # ChatOpenAI 객체 초기화
    chat_model = ChatOpenAI(
        model="gpt-4",  # OpenAI 모델 이름
        openai_api_key=api_key,  # OpenAI API 키
        temperature=0.7,  # 창의성 조정
        max_retries=3,  # 요청 실패 시 재시도 횟수
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False,
    )

    response = chain({"question": user_input, "chat_history": st.session_state.chat_history})
    ai_response = response["answer"]

except Exception as e:
    ai_response = f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}"

# AI 응답 기록 추가 및 출력
st.session_state.chat_history.append(("튜터링", ai_response))
st.chat_message("assistant").markdown(ai_response)

# 기존 대화 히스토리 표시
for role, content in st.session_state.chat_history:
    if role == "사용자":
        st.chat_message("user").markdown(content)
    else:
        st.chat_message("assistant").markdown(content)
