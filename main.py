import os

from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = f.read()
    return data



def load_and_split_files(files):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    docs = [markdown_splitter.split_text(load_data(file_path)) for file_path in files]

    return docs

def main(files):
    docs = load_and_split_files(files)
    print(len(docs))




if __name__ == "__main__":
    files_path = "data/pages"
    files = [dirpath + "/" + f for dirpath,_,filenames in os.walk(files_path) for f in filenames]
    main(files)