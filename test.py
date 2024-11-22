import streamlit as st
import pandas as pd
import numpy as np

st.title("스파르타코딩클럽 AI 8기 예제")
st.header("지금 스트림릿 예제를 만들고 있어요.")
st.text("follow me")

st.markdown("### H3 글씨를 표현합니다.")
st.latex("E=mc^2") # 수식은 고정 텍스트 기울임체가 자동 적용됨

if st.button("버튼을 클릭하세요") :
    st.write("버튼이 눌렸습니다!")

agree_box = st.checkbox("동의하시겠습니까?")
if agree_box is True:
    st.write("동의하셨습니다!")

volume = st.slider("음악 볼륨", 0, 100, 50)
st.write("음악 볼륨은 " + str(volume) + "입니다.")

gender = st.radio("성별", ["남자", "여자"])
st.write("성별은 " + gender + "입니다.")

flower = st.selectbox("좋아하는 꽃", ["해바라기", "장미", "튤립", "유채꽃"])

df = pd.DataFrame({
    "학번": ["20170321", "20180111", "20192020", "20200589"],
    "이름": ["김철수", "최영희", "신지수", "이철주"]
})
# st.dataframe(df)
st.container(border=False, height=30) #위아래 빈공간 보더(테두리) 없이설정
st.table(df)

# chart_data = pd.DataFrame( np.random.randn(20, 3), columns=["a", "b", "c"] ) 
# st.line_chart(chart_data) # 랜덤 꺾은 선 그래프

chart_data = pd.DataFrame({
    "국어": [100, 95, 80],
    "영어": [80, 95, 100],
    "수학": [95, 100, 80]
}) # 점수 지정도 가능함
# st.line_chart(chart_data) # 라인 차트
st.bar_chart(chart_data) # 바 차트
