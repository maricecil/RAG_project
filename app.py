import streamlit as st
import numpy as np

st.title("간단한 숫자 데이터 분석하기")

# 사용자로부터 숫자 입력받기
numbers = st.text_input("숫자 리스트를 입력하세요 (쉼표로 구분)", "1,2,3,4,5")  # 플레이스홀더, 기본값
number_list = [float(x) for x in numbers.split(",")]

# 통계 정보 계산
mean_value = np.mean(number_list) # 평균
median_value = np.median(number_list) # 중앙값
stdev_value = np.std(number_list) # 표준편차

# 결과 출력
st.write(f"평균값: {mean_value}") # st는 streamlit 약자임
st.write(f"중앙값: {median_value}")
st.write(f"표준편차: {stdev_value}")