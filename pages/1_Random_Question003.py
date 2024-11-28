import streamlit as st
import os
import random
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import ScriptRequestQueue

# PDF 매핑 (PDF 경로 정의)
CATEGORY_PDF_MAPPING = {
    "python": "C:/Users/user/RAG_project/JSW/QnA/python.pdf",
    "machine_learning": "C:/Users/user/RAG_project/JSW/QnA/machine_learning.pdf",
    "deep_learning": "C:/Users/user/RAG_project/JSW/QnA/deep_learning.pdf",
    "data_structure": "C:/Users/user/RAG_project/JSW/QnA/data_structure.pdf",
    "operating_system": "C:/Users/user/RAG_project/JSW/QnA/operating_system.pdf",
    "network": "C:/Users/user/RAG_project/JSW/QnA/network.pdf",
    "statistics": "C:/Users/user/RAG_project/JSW/QnA/statistics.pdf",
    "algorithm": "C:/Users/user/RAG_project/JSW/QnA/algorithm.pdf",
}

# 세션 상태 초기화 함수
def initialize_session_state():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = None  # API 키 저장
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = list(CATEGORY_PDF_MAPPING.keys())[0]  # 기본 카테고리
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # 대화 기록 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # 질문/답변 초기화

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

# 페이지 설정
st.set_page_config(page_title="무작위 질문")

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
        index=list(CATEGORY_PDF_MAPPING.keys()).index(st.session_state["selected_category"]),
    )
    st.session_state["selected_category"] = selected_category

# 메인 화면
st.markdown("## 무작위 질문 생성")
st.info("📚 이 세션에서는 AI 면접관이 사용자가 선택한 과목에 대해 면접 질문을 생성하고 사용자의 대답을 평가합니다.")

# 질문 생성 버튼
st.write("🎯 면접 질문")
if not st.session_state["messages"]:
    st.info("👇 아래 버튼을 눌러 첫 면접 질문을 생성해보세요!")
else:
    st.info("💡 새로운 질문을 생성하려면 아래 버튼을 클릭하세요")

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

            previous_questions = [msg["question"] for msg in st.session_state.get("messages", [])][-5:]
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
                    st.experimental_rerun()
                else:
                    st.warning("모든 질문을 완료했습니다. 다른 카테고리를 선택해주세요.")
            else:
                st.warning("PDF에서 백틱(`)으로 둘러싸인 질문을 찾을 수 없습니다.")
        else:
            st.error("PDF에서 텍스트를 추출할 수 없습니다.")
    except Exception as e:
        st.error(f"PDF 파일 읽기 오류: {str(e)}")
        st.info("PDF 파일이 존재하고 읽기 가능한지 확인해주세요.")

# 사용자 입력
st.markdown("### 면접 질문에 대한 응답을 입력하세요:")
user_message = st.text_input("메시지를 입력하세요:", placeholder="여기에 메시지를 입력하세요...")

# 전송 버튼
if st.button("전송"):
    if user_message.strip():
        st.session_state["chat_history"].append(f"사용자: {user_message}")
    else:
        st.warning("빈 메시지는 입력할 수 없습니다.")

# 채팅 기록 표시
st.markdown("### 채팅 기록")
for message in st.session_state["chat_history"]:
    st.write(message)
