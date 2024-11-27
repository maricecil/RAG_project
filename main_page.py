import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# 페이지 설정
im = Image.open("assistent.png")
st.set_page_config(page_title="AI Interviewer", layout="centered", page_icon=im)

# 언어 선택
lan = st.selectbox("#### Language", ["English", "한국어"])

# English 설정 시
if lan == "English":
    home_title = "AI Interviewer"
    home_introduction = "Welcome to AI Interviewer, empowering your interview preparation with generative AI."

    # 사이드바 구성
    with st.sidebar:
        st.markdown('AI Interviewer')
        st.selectbox('#### choice', ['Python', 'Machine_Learning', 'Deep_Learning', 'Network', 'Statistics', 
                                     'Operating_System', 'Data_Structure', 'Algorithm'])
        openai_api_key = st.text_input('OpenAI API KEY', key="chatbot_api_key", type="password")
        process = st.button("Process")
        st.markdown(""" 
        #### Powered by
        [OpenAI](https://openai.com/)
        [AI Tech Interview](https://boostdevs.gitbook.io/ai-tech-interview)
        [FAISS](https://github.com/facebookresearch/faiss)
        [Langchain](https://github.com/hwchase17/langchain)
        """)

    st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.image(im, width=100)
    st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""", unsafe_allow_html=True)

    st.markdown("Welcome to AI Interviewer! 👏 AI Interviewer is your personal interviewer powered by generative AI that conducts mock interviews."
                "You can upload your resume and enter job descriptions, and AI Interviewer will ask you customized questions. Additionally, you can configure your own Interviewer!")
    
    with st.expander("Updates"):
        st.write("""
        08/13/2023
        - Fix the error that occurs when the user input fails to be recorded.
        """)

    with st.expander("What's coming next?"):
        st.write("""
        Improved voice interaction for a seamless experience.
        """)

    st.markdown("#### Get started!")
    st.markdown("Select one of the following screens to start your interview!")
    
    # 메뉴 옵션
    selected = option_menu(
        menu_title=None,
        options=["Select Question", "Provide Question"],
        icons=["cast", "cloud-upload"],
        default_index=0,
        orientation="horizontal"
    )

    if selected == 'Select Question':
        st.info("""
        📚In this session, an AI interviewer generates interview questions on a topic of your choice and evaluates candidate responses.
        Notes: The maximum length of an answer is 4097 tokens!
        - Tutor and give feedback on your answers.
        - To start a new session, simply refresh the page.
        - Choose a subject, get started, and have fun!
        """)
        # if st.button("Start Interview!"):
        #     switch_page("Select Question Screen")
        #     pass

    if selected == 'Provide Question':
        st.info("""
        📚In this session, you'll select a topic presented by an AI interviewer and evaluate your responses.
        Notes: The maximum length of an answer is 4097 tokens!
        - You'll receive tutoring and feedback on your answers.
        - To start a new session, simply refresh the page.
        - Choose a topic, get started, and have fun!
        """)
        # if st.button("Start Interview!"):
        #     switch_page("Provide Question Screen")
        #     pass

    st.markdown("#### Wiki")
    st.write('[Click here to view common FAQs, future updates and more!](https://jiatastic.notion.site/wiki-8d962051e57a48ccb304e920afa0c6a8?pvs=4)')

# 한국어 설정 시
if lan == "한국어":
    home_title = "AI 면접관"
    home_introduction = "AI로 면접 준비를 도와주는 AI Interviewer에 오신 것을 환영합니다."

    # 사이드바 구성
    with st.sidebar:
        st.markdown('AI Interviewer')
        st.selectbox('#### choice', ['Python', 'Machine_Learning', 'Deep_Learning', 'Network', 'Statistics', 
                                     'Operating_System', 'Data_Structure', 'Algorithm'])
        openai_api_key = st.text_input('OpenAI API KEY', key="chatbot_api_key", type="password")
        process = st.button("Process")
        st.markdown(""" 
        #### Powered by
        [OpenAI](https://openai.com/)
        [AI Tech Interview](https://boostdevs.gitbook.io/ai-tech-interview)
        [FAISS](https://github.com/facebookresearch/faiss)
        [Langchain](https://github.com/hwchase17/langchain)
        """)

    st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.image(im, width=100)
    st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""", unsafe_allow_html=True)

    st.markdown("AI 면접관에 오신 것을 환영합니다! AI 면접관은 생성형 인공 지능으로 구동되는 개인 면접관으로 모의 면접을 진행합니다. 이력서를 업로드하거나 직무 설명을 복사하여 붙여넣으면 AI 면접관이 맞춤형 질문을 합니다.")
    
    with st.expander("업데이트 로그"):
        st.write("""
        08/13/2023
        - 사용자 입력 실패 시 오류 수정
        """)

    with st.expander("향후 계획"):
        st.write("""
        - 보다 안정적이고 빠른 음성 상호 작용 제공
        - 한국어 모의 면접 지원
        """)

    st.markdown("#### 시작해 보겠습니다!")
    st.markdown("다음 중 하나를 선택하여 인터뷰를 시작하세요！")

    # 메뉴 옵션
    selected = option_menu(
        menu_title=None,
        options=["Select Question", "Provide Question"],
        icons=["cast", "cloud-upload"],
        default_index=0,
        orientation="horizontal"
    )

    if selected == 'Select Question':
        st.info("""
        📚이 세션에서는 AI 면접관이 사용자가 선택한 과목에 대하여 면접 질문을 생성하고 사용자의 대답을 평가합니다.
        참고: 답변의 최대 길이는 4097토큰입니다!
        - 사용자 대답을 튜터링 및 피드백을 합니다.
        - 새 세션을 시작하려면 페이지를 새로고침하면 됩니다.
        - 과목을 선택하고 시작해 즐겨보세요!
        """)
        # if st.button("면접 시작!"):
        #     switch_page("질문 페이지")
        #     pass

    if selected == 'Provide Question':
        st.info("""
        📚 이 세션에서는 AI 면접관이 제시한 주제를 선택하고 지원자의 답변을 평가합니다.
        참고: 답변의 최대 길이는 4097토큰입니다!
        - 답변에 대한 튜터링과 피드백을 받을 수 있습니다.
        - 새 세션을 시작하려면 페이지를 새로고침하기만 하면 됩니다.
        - 주제를 선택하고 시작하여 재미있게 즐겨보세요!
        """)
        # if st.button("면접 시작!"):
        #     switch_page("질문 페이지")
        #     pass

