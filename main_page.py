import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from utils import switch_page
import pathlib
import json

# 페이지 설정
st.set_page_config(page_title="면접 튜터링", layout="centered")

# 사용자 데이터 경로부터 사용자 데이터 저장 함수까지는 다음 세션에서nst.session_state["authenticated"]를 사용하기 위해 정의했습니다.

# 사용자 데이터 경로
users_file_path = pathlib.Path("users.json")

# 사용자 데이터 로드 함수
def load_users():
    
    if users_file_path.exists():
        with open(users_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 사용자 데이터 저장 함수
def save_users(users):
    with open(users_file_path, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# 로그인 처리
if not st.session_state["authenticated"]:
    # 문구 추가
    st.markdown("### 로그인 또는 회원가입이 필요합니다.")
    tab1, tab2 = st.tabs(["로그인", "회원가입"])

    # 사용자 데이터 로드
    users = load_users()

    # 로그인 화면
    with tab1:
        login_id = st.text_input("사용자 ID:", key="login_id")
        login_pw = st.text_input("비밀번호:", type="password", key="login_pw")
        if st.button("로그인", key="login_btn"):
            if login_id in users and users[login_id] == login_pw:
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = login_id
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 일치하지 않습니다.")

    with tab2:
        new_id = st.text_input("새로운 ID:", key="new_id")
        new_pw = st.text_input("새로운 비밀번호:", type="password", key="new_pw")
        new_pw_confirm = st.text_input("비밀번호 확인:", type="password", key="new_pw_confirm")
        if st.button("회원가입", key="register_btn"):
            if not new_id or not new_pw:
                st.error("ID와 비밀번호를 모두 입력해주세요.")
            elif new_pw != new_pw_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                if new_id in users:
                    st.error("이미 존재하는 ID입니다.")
                else:
                    users[new_id] = new_pw
                    save_users(users)  # 사용자 데이터 저장
                    st.success("회원가입이 완료되었습니다. 로그인해주세요.")

    st.stop()

image = Image.open("AI_logo.png")
st.sidebar.image(image, use_container_width=True)

# 로그아웃 버튼 처리
st.sidebar.markdown(f"**사용자: {st.session_state['user_id']}**")
if st.sidebar.button("로그아웃", key="logout_btn"):
    st.session_state["authenticated"] = False
    st.session_state["user_id"] = None
    st.rerun()

# 외국어 서비스를 추가하지 않게 된다면 이 부분과 아래도 수정하기
lan = st.radio("Language", ["한국어"], horizontal=True)

# 한국어 설정 시
if lan == "한국어":
    home_title = "AI 웹 개발자 면접 튜터링"
    home_introduction = "AI로 면접 준비를 도와주는 AI 웹 개발자 면접 튜터링에 오신 것을 환영합니다."

    st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.image(image, width=100)
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
            switch_page("main8")


    if selected == '질문 제공':
        st.info("""
        📚 이 세션에서는 AI 면접관이 제시한 주제를 선택하고 지원자의 답변을 평가합니다.
        참고: 답변의 최대 길이는 4097토큰입니다!
        - 답변에 대한 튜터링과 피드백을 받을 수 있습니다.
        - 새 세션을 시작하려면 페이지를 새로고침하기만 하면 됩니다.
        - 주제를 선택하고 시작하여 재미있게 즐겨보세요!
        """)
        if st.button("공부 시작!"):
            switch_page("공부방")