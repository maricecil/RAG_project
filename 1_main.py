# -*- coding: utf-8 -*-

# 1. 프로그램에 필요한 도구들을 가져옵니다
import os  # 파일/폴더 관리를 위한 라이브러리
import streamlit as st  # 웹 애플리케이션을 만드는 라이브러리
from streamlit_option_menu import option_menu
import pathlib # 파일 경로를 더 쉽게 다루는 도구
import streamlit as st  # 웹 애플리케이션을 만드는 라이브러리
import base64 # 이진 데이터를 텍스트로 변환하기 위해 사용 
from PIL import Image
from utils import switch_page
import json

# 웹페이지 제목 설정
st.set_page_config(page_title="AI Tutor")

# CSS 스타일링 추가
st.markdown(
    """
    <style>
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%; /* 부모 컨테이너의 너비에 맞춤 */
        max-width: 350px; /* 최대 너비 제한 */
    }
    
    body {
        background-color: #071952; /* 원하는 배경 색상 */
        color: #FFFFFF; /* 텍스트 색상 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 이미지 로드 및 표시 (HTML 사용)
im = Image.open("icon.png")

# 이미지를 base64로 인코딩하여 HTML로 삽입
with open("로고디자인11.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()
st.markdown(
    f'<img src="data:image/png;base64,{encoded_image}" class="centered-image">',
    unsafe_allow_html=True,
)

# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 사용자 데이터 경로부터 사용자 데이터 저장 함수까지는 다음 세션에서nst.session_state["authenticated"]를 사용하기 위해 정의했습니다.
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

# 로그인 처리
if not st.session_state.get("authenticated", False):  # 안전하게 기본값 설정
    # 문구 추가
    st.markdown(
        """
        <p style="color:white; font-weight:bold; font-size:20px;">
            AI 튜터링 서비스에 오신 것을 환영합니다!
        </p>
        <p style="color:gray; font-size:16px;">
            본 서비스는 간단한 로그인 절차를 통해 시작할 수 있으며,  <br>
            총 8개 분야의 면접 질문들이 준비되어 있습니다. <br>
            질문에 대해 답변하면 AI 면접관이 평가하고 피드백을 제공합니다!
        </p>
        """, 
        unsafe_allow_html=True  # 이 부분은 따로 전달
    )

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

# 로그아웃 버튼 처리
st.sidebar.markdown(f"**사용자: {st.session_state['user_id']}**")
if st.sidebar.button("로그아웃", key="logout_btn"):
    st.session_state["authenticated"] = False
    st.session_state["user_id"] = None
    st.rerun()

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

# 서비스 이름은 가제이므로 수정될 수 있음
st.markdown(
    """
    <p style="color:white; font-weight:bold; font-size:20px;">
        AI 면접 튜터링 서비스에 오신 것을 환영합니다!
    </p>
    <p style="color:gray; font-size:16px;">
        본 서비스는 사용자가 질문에 답하면 그에 대한 피드백이 제공됩니다.<br>
        좋은 점, 보완점, 추가로 언급할 내용, 그리고 개선된 답변의 예시를 드립니다.<br>
    </p>
    """, 
    unsafe_allow_html=True
)
with st.expander("향후 계획"): # expander와 여러줄 쓰기를 통해 향후 계획을 제시
    st.write("""
    - 피드백 속도 향상 및 다국어 지원
    - 새로운 면접 카테고리 추가
    - PDF 파일 정기 업데이트
    - 사용자 피드백 수집
    - 지속적 모니터링을 통한 성능 최적화
    """)
    
selected = option_menu(
            menu_title= None,  # 메뉴의 제목 (없으면 메뉴만 표시됨)
            options=["면접실", "자습실"], # 메뉴에서 선택할 옵션 리스트
            icons = ["chat-left-dots", "chat-left-text-fill"], # 옵션 옆에 표시할 아이콘 리스트
            default_index=0, # 기본으로 선택된 옵션의 인덱스 (0부터 시작)
            orientation="horizontal",) # 메뉴 방향 (수평(horizontal) 또는 수직(vertical))

if selected == '면접실':
    st.info(
        """
        📚 면접을 시작하려면:
        - 좌측의 **면접실 버튼**이나 하단의 **면접 시작 버튼**을 누르세요!
        - **주제를 선택**하고, 새로운 질문 생성 버튼을 누르면 해당 분야의 질문을 받을 수 있습니다.
        - AI 튜터가 사용자의 답변을 분석하고 **상세한 피드백**을 제공합니다!
        """
    )
    
    if st.button("면접 시작!"):
        switch_page("면접실")
        
# 사이드바에 아이콘 추가
icon_path = "로고1.png"  # 아이콘 이미지 파일 경로
if os.path.exists(icon_path):  # 파일 존재 여부 확인
    # 이미지와 링크를 함께 추가
    st.sidebar.markdown(
        f"""
        <a href="https://boostdevs.gitbook.io/ai-tech-interview" target="_blank">
            <img src="data:image/png;base64,{base64.b64encode(open(icon_path, 'rb').read()).decode()}" width="100%">
        </a>
        <p style="text-align: center; color: gray; font-size: 12px;">
            자료는 ai-tech-interview 를 기반으로 작성되었습니다.
            로고를 클릭하면 사이트로 이동합니다.
        </p>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("아이콘 파일을 찾을 수 없습니다.")  # 경고 메시지 출력

if selected == '자습실':
    st.info(
        """
        \U0001F4DA 학습을 시작하려면:
        - 좌측의 **자습실 버튼**이나 하단의 **학습 시작 버튼**을 누르세요!
        - **주제를 선택**하고 질문을 선택하면 해당 주제에 대한 학습이 시작됩니다.
        - AI 튜터가 사용자의 답변을 분석하고 **모범 답안**을 제공합니다!
        """.encode('utf-8').decode('utf-8')
    )
    
    if st.button("학습 시작!".encode('utf-8').decode('utf-8')):
        switch_page("자습실")