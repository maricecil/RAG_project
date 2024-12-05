import streamlit as st
from PIL import Image
import time
from datetime import datetime
import webbrowser
import login
st.markdown(
    """<style>
    .stApp {
        background-color: #222831;
        height: 100vh;
        margin: 0;
    }
    .stImage img {
        border-radius: 15px;         
        border: 2px solid #00000; 
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5); 
        max-width: 100%;                      
        display: block;
        margin: auto;
        width: 800px;
        height: 250px;
    }
    .stSidebar {
        background-color: #31363F;
        width: 300px;
    }
    .stText{
        
    }
    .stSelectbox {
        background-color: #EBEAFF;
        border: 2px solid #00000;
        padding: 10px;
        border-radius: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# def run():
image = Image.open("C:/Users/kevinkim/Downloads/AI_logo.png")
st.sidebar.image(image, use_container_width=True)

current_date_space = st.sidebar.empty() # 빈 공간을 생성
current_time_space = st.sidebar.empty() # 빈 공간을 생성
st.markdown(f'<p style="color:white; font-weight:bold; font-size : 20px">반가워요! 저는 IT 면접 시스템을 도와줄 AI입니다!</p>', unsafe_allow_html = True)
# st.sidebar.markdown('<p style="color:white; font-weight:bold; font-size:20px;">주제를 골라주세요!</p>', unsafe_allow_html=True)
topic = ["파이썬", "머신러닝", "딥러닝", "데이터구조", "운영체제", "네트워크", "통계", "알고리즘",]
st.sidebar.selectbox('원하시는 주제를 골라주세요!', [topics for topics in topic])
# progress = st.progress(0)
    # for i in range(101):
    #     time.sleep(0.05)
    #     progress.progress(i)
while True:
    current_day = datetime.now().strftime('%Y-%m-%d') # 현재 시간 포맷팅
    current_time = datetime.now().strftime('%H:%M:%S') # 현재 시간 포맷팅
    
    current_date_space.markdown(f'<p style="color:white; font-weight:bold;">현재 날짜 - {current_day}</p>', unsafe_allow_html=True)
    current_time_space.markdown(f'<p style="color:white; font-weight:bold;">현재 시간 - {current_time}</p>', unsafe_allow_html=True)
    time.sleep(1) # 1초 간격으로 갱신 

