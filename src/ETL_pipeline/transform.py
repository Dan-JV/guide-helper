import json
import os
from typing import Tuple

import polars as pl
from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader


def save_splits(pages, name="pages"):
    if not os.path.exists(f'data/{name}'):
        os.makedirs(f'data/{name}')

    for i, page in enumerate(pages):
        if page.strip():  # Ensure the page is not just whitespace
            output_file_path = f'data/pages/page_{i + 1}.txt'
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(page.strip())

    print("pages saved sucessfully")

def split_data(data):
    splits = data.split('----------')
    save_splits(pages=splits, name="pages")

    return splits 

def preprocess_chunks(chunks, metadata, split_id=0):
    url = chunks[0].page_content

    # we start from 1 because the first chunk is the url
    for chunk_id, chunk in enumerate(chunks[1:]):
        chunk.metadata["url"] = url
        chunk_dict = {"metadata": chunk.metadata, "content": chunk.page_content}

        chunk_metadata: Tuple = metadata.filter(pl.col("url") == url).row(0)
        chunk_dict["metadata"]["description"] = chunk_metadata[1]
        chunk_dict["metadata"]["primary_keywords"] = chunk_metadata[2]
        chunk_dict["metadata"]["Slug"] = chunk_metadata[4]

        output_file_path = f'data/chunks/chunk_{split_id}_{chunk_id}.json'  # Changed file extension to .json
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(chunk_dict, output_file, ensure_ascii=False)
    print(f"chunked page {split_id} sucessfull")


def create_chunks_from_splits(splits, metadata):
    if not os.path.exists('data/chunks'):
        os.makedirs('data/chunks')

    # ---------------------------------#
    # Chunking config options
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    chunk_size = 1024
    chunk_overlap = 64
    #----------------------------------#

    # Markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    # Recursive splitter that splits the large markdown into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for split_id, split in enumerate(splits):
        md_header_splits = markdown_splitter.split_text(split)
        chunks = text_splitter.split_documents(md_header_splits)

        # add metadata to chunks
        preprocess_chunks(chunks, metadata, split_id)
    
def create_docs():
    reader = SimpleDirectoryReader(input_dir="data/chunks", recursive=True)
    docs = list()

    for chunk in reader.iter_data():
        chunk = chunk[0]
        text = chunk.text
        text_dict = json.loads(text)
        metadata = text_dict['metadata']
        page_content = text_dict['content']
        doc = Document(metadata=metadata, page_content=page_content)
        docs.append(doc)

    return docs



def transform(data, metadata):
    splits = split_data(data)
    create_chunks_from_splits(splits, metadata)
    docs = create_docs()

    return docs
