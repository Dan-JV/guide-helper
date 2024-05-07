import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_data():
    with open("data/support_articles_raw.txt", encoding="utf-8") as f:
        data = f.read()
    
    return data


text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "https",
        "#",
        "##",
        "###"
    ],
    # Existing args
)

data = load_data()


pages = html_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

for page in pages:
    chunks = text_splitter.create_documents([data])

    for chunk in chunks:
        chunk.metadata["url"] = 





print(len(texts))
print(texts[0])
print(texts[1])