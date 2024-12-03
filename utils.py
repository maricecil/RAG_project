from streamlit.runtime.scriptrunner import RerunException, RerunData
from streamlit.source_util import get_pages

def switch_page(page_name: str):
    """
    페이지를 전환하는 함수.
    """
    def standardize_name(name: str) -> str:
        # 페이지 이름을 표준화: 소문자로 변환하고 언더스코어를 공백으로 치환
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)
    pages = get_pages("1_main.py")

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(RerunData(page_script_hash=page_hash))

    available_pages = [standardize_name(config["page_name"]) for config in pages.values()]
    raise ValueError(f"Could not find page {page_name}. Available pages: {available_pages}")