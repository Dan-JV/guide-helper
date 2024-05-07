import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_data():
    with open("data/support_articles_raw.txt", encoding="utf-8") as f:
        data = f.read()
    
    return data


data = load_data()


text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "https",
        "#",
        "##",
        "###"
    ],
    # Existing args
)

texts = text_splitter.create_documents([data])
print(len(texts))
print(texts[0])
print(texts[1])