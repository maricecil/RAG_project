import streamlit as st
from utils import switch_page

st.title("Provide Question Page")
st.write("This is the Provide Question page.")

# 돌아가기 버튼
if st.button("Go Back"):
    switch_page("main_page")
