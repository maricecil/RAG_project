import streamlit as st
from utils import switch_page

st.title("질문 제공 페이지")
st.write("이곳은 질문 제공 페이지입니다.")

if st.button("뒤로가기"):
    switch_page("main_page")
