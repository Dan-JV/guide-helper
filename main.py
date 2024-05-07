import os
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter


headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = f.read()
    return data



def split_page(files):
    for file in files:
        # counter for page
        i = 0

        page = load_data(file)     

        chunks = markdown_splitter.split_text(page)

        url = chunks[0]

        print(len(chunks))

        for chunk in chunks:
            # Counter for chunks
            j = 0

            try:
                # First chunk is the url
                chunk.metadata["url"] = url
            except:
                print(f"Failed to add url to chuck for {file}")

            json_chunk = json.loads(chunk.json())

            # Assuming `chunk` is a Python dictionary that you want to serialize into JSON
            output_file_path = f'data/chunks/chunk_{i}_{j}.json'  # Changed file extension to .json
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(json_chunk, output_file)  # Serializes `chunk` to a JSON formatted string and writes it to `output_file`


    return chunks

def main(files):
    split_page(files)




if __name__ == "__main__":
    files_path = "data/pages"
    files = [dirpath + "/" + f for dirpath,_,filenames in os.walk(files_path) for f in filenames]
    main(files)