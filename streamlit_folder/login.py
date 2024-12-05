import streamlit as st
import ai_page

# # 세션 상태 초기화
# if 'current_page' not in st.session_state:
#     st.session_state.current_page = "login"  # 기본 페이지: 로그인

# # 로그인 버튼 처리
# if st.session_state.current_page == "login":
#     if st.button('로그인하기'):
#         st.session_state.current_page = "ai_page"  # AI 페이지로 전환

# # 페이지 렌더링
# if st.session_state.current_page == "ai_page":
#     ai_page.run()
