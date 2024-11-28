import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from utils import switch_page

# 페이지 설정
im = Image.open("assistent.png")
st.set_page_config(page_title="면접 튜터링", layout="centered", page_icon=im)

# 외국어 서비스를 추가하지 않게 된다면 이 부분과 아래도 수정하기
lan = st.radio("Language", ["한국어"], horizontal=True)

# 한국어 설정 시
if lan == "한국어":
    home_title = "AI 웹 개발자 면접 튜터링"
    home_introduction = "AI로 면접 준비를 도와주는 AI 웹 개발자 면접 튜터링에 오신 것을 환영합니다."

    st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.image(im, width=100)
    st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>""", unsafe_allow_html=True)

    st.markdown("AI 웹 개발자 면접 튜터링에 오신 것을 환영합니다! AI 개인 면접관으로 모의 면접을 진행합니다. 질문 유형과 직무 분야를 고르면 AI 면접관이 면접 예상 질문을 합니다.")

    with st.expander("향후 계획"):
        st.write("""
        - 보다 안정적이고 빠른 피드백 및 음성 상호 작용 제공
        """)

    st.markdown("#### 시작해 보겠습니다!")
    st.markdown("다음 중 하나를 선택하여 인터뷰를 시작하세요！")

    # 메뉴 옵션, 나중에 아이콘 변경하기
    selected = option_menu(
        menu_title=None,
        options=["무작위 질문", "질문 제공"],
        icons=["cast", "cloud-upload"],
        default_index=0,
        orientation="horizontal"
    )

    # 이 부분도 나중에 수정하기
    if selected == '무작위 질문':
        st.info("""
        📚이 세션에서는 AI 면접관이 사용자가 선택한 과목에 대하여 면접 질문을 생성하고 사용자의 대답을 평가합니다.
        참고: 답변의 최대 길이는 4097토큰입니다!
        - 사용자 대답을 튜터링 및 피드백을 합니다.
        - 새 세션을 시작하려면 페이지를 새로고침하면 됩니다.
        - 과목을 선택하고 시작해 즐겨보세요!
        """)
        if st.button("면접 시작!"):
            switch_page("랜덤 질문")


    if selected == '질문 제공':
        st.info("""
        📚 이 세션에서는 AI 면접관이 제시한 주제를 선택하고 지원자의 답변을 평가합니다.
        참고: 답변의 최대 길이는 4097토큰입니다!
        - 답변에 대한 튜터링과 피드백을 받을 수 있습니다.
        - 새 세션을 시작하려면 페이지를 새로고침하기만 하면 됩니다.
        - 주제를 선택하고 시작하여 재미있게 즐겨보세요!
        """)
        if st.button("면접 시작!"):
            switch_page("랜덤_질문")