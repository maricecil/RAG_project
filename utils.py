from streamlit.runtime.scriptrunner import RerunException, RerunData
from streamlit.source_util import get_pages

def switch_page(page_name: str):
    """Switch to a specific page based on the page name."""
    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    # Get all available pages
    pages = get_pages("main_page.py")  # 메인 파일 이름을 입력하세요.

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(RerunData(page_script_hash=page_hash))

    available_pages = [standardize_name(config["page_name"]) for config in pages.values()]
    raise ValueError(f"Could not find page {page_name}. Available pages: {available_pages}")
