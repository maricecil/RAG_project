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
from datetime import datetime # 날짜와 시간을 다루는 도구
import pathlib # 파일 경로를 더 쉽게 다루는 도구
import streamlit as st  # 웹 애플리케이션을 만드는 라이브러리
from PIL import Image
from utils import switch_page

# 웹페이지의 제목을 설정합니다 # streamlit title
st.set_page_config(page_title = "AI Tutor")
im = Image.open("Icon.png")
st.image(im, width=350)
st.title("**AI 개발자 면접 튜터링**")

# 서비스 이름은 가제이므로 수정될 수 있음
st.markdown(
    """
    <p style="color:gray; font-weight:bold; font-size:20px;">
        AI 튜터링 서비스에 오신 것을 환영합니다!
    </p>
    <p style="color:green; font-size:16px;">
        본 서비스는 간단한 로그인 절차를 통해 시작할 수 있으며,  <br>
        총 8개 분야의 면접 질문들이 준비되어 있습니다. <br>
        질문에 대해 답변하면 AI 면접관이 평가하고 피드백을 제공합니다!
    </p>
    """, 
    unsafe_allow_html=True
)
with st.expander("향후 계획"):
    st.write("피드백 속도 향상 및 영어 기능 지원")
    
st.info("면접을 시작하려면 좌측의 interview 버튼을 눌러 로그인 한뒤, 버튼을 눌러 질문을 받으세요!")

st.audio("bgm001.mp3", format="audio/mpeg", loop=True)