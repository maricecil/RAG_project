import streamlit as st
import tiktoken #텍스트를 청크로 나눌 때 문자의 갯수를 무엇을 기준으로 산정할지, 토큰갯수를 새기위한 라이브러리
from loguru import logger #어떤 행동을 취했을 때, streamlit 웹사이트 상에서 구동한 것이 로그로 남도록 하기 위한 것

#ConversationalRetrievalChain은 메모리는 가지고 있어서, 이걸 구현할려면 ConversationBufferMemory가 필요(몇개까지 저장할 것인지 결정하는 부분)
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveJsonSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

#밑 두 라이브러리는 전부 메모리를 구현하기 위한것
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# st.set_page_config(
#     page_title = 'DirChat',
#     page_icon = ':car:'
# )
# st.title('_Today IT Learning :red[IT]_ :books:') #_을 붙이면 조금 기울어짐

# if st.button('버튼'):
#     st.write('안녕')
# # 체크박스
# agree = st.checkbox('동의')
# if agree:
#     st.write('동의완료')
    
# # 슬라이더
# age = st.slider('나이', 0, 100)
# st.write(f'선택한 나이는 {age}입니다.')

# volume= st.slider('음량', 0, 100)

# gender = st.radio('성별', ['남자', '여자', '없음']) #선택 사항을 만들 때는 무조건 []로 감싸기
# st.write(f'성별은 {gender}입니다')

# flower = st.selectbox('꽃', ['1', '2', '3'])
# if flower == '1':
#     st.write('1')
    
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")
# # with tab1:
# #     st.radio("Select one:", [1, 2])
    
    
# expand = st.expander("My label", icon=":material/info:")
# expand.write("Inside the expander.")
# pop = st.popover("Button label")
# pop.checkbox("Show all")

# # You can also use "with" notation:
# with expand:
#     st.radio("Select one:", [1, 2])
    
#st.dataframe(my_dataframe)
#st.table(data.iloc[0:10])
st.json({"foo":"bar","fu":"ba"})
st.metric("My metric", 42, 6)
