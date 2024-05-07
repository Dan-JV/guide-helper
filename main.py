from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = f.read()
    return data

def main(file_path):
    data = load_data(file_path)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    docs = markdown_splitter.split_text(data)



if __name__ == "__main__":
    file_path = "data/support_articles_raw.txt"
    main(file_path)