import streamlit as st
import numpy as np

def my_llm_stream():
    yield "Response part 1"
    yield "Response part 2"
    
gen = my_llm_stream()
st.write(next(gen))

if st.button('버튼을 눌러주세요'):
    st.write(next(gen))

def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

for num in count_up_to(5):
    st.write(num)

col1, col2 = st.columns(2)
col1.write("Column 1 content")
col2.write("Column 2 content")

col1, col2, col3 = st.columns([8,1,1])
col1.write("Large column")
col2.write("Small column")
col3.write("Small column")

col1, col2 = st.columns(2, vertical_alignment="bottom")
col1.write("This is aligned at the bottom")
col2.write("This is also aligned at the bottom")

choose = st.radio("Choose one", [1, 2])
with col1:
    choose
    if choose == 1:
        st.write('hi')
        
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Content of Tab 1")
with tab2:
    st.write("Content of Tab 2")
    
with st.sidebar:
    st.radio("Select an option", [1, 2])
    st.slider("Pick a number", 0, 100)

with tab1:
    st.write("This is tab 1 content")
pop = st.popover("Click me for more info")
pop.checkbox("Show details")

@st.dialog("Welcome!")
def modal_dialog():
    st.write("Hello, welcome to our app!")

st.feedback("thumbs")
st.pills("Tags", ["Sports", "Politics"])
st.segmented_control("Filter", ["Open", "Closed"])
st.toggle("Enable")
st.multiselect("Buy", ["milk", "apples", "potatoes"])
st.select_slider("Pick a size", ["S", "M", "L"])

slider_value = st.slider("Pick a number", 0, 100, key="unique_slider_1")
st.write("Selected number:", slider_value)
st.selectbox("Pick one", ["cats", "dogs"])
st.radio("Pick one", ["cats", "dogs"])
if st.checkbox("I agree"):
    st.write("You agreed!")
st.text_input("First name")
st.number_input("Pick a number", 0, 10)
st.text_area("Text to translate")

st.date_input("Your birthday")
st.time_input("Meeting time")
st.file_uploader("Upload a CSV")
st.audio_input("Record a voice message")
st.camera_input("Take a picture")
st.color_picker("Pick a color")

my_slider_val = st.slider("Quinn Mallory", 1, 88)
st.write(my_slider_val)
st.slider("Pick a number", 0, 100, disabled=True, key = 'hithere')

with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

st.chat_input("Say something")

with st.container():  
    st.chat_input("Say something", key = 'another') 

import pandas as pd

# 데이터프레임 초기화
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

# 데이터 표시 및 행 추가
element = st.dataframe(df1)
element.add_rows(df2)  # df2 데이터를 df1 뒤에 추가

with st.echo():
    st.write("This will be displayed on the app and executed.")
    
elements = st.container()  # 컨테이너 생성
st.write("Hello")          # "Hello" 출력

st.toast("Warming up...")
st.snow()
st.metric("Temperature", "23°C", "-2°C")