

from FlagEmbedding import BGEM3FlagModel


def embed_chunks(chunks, model):
    embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk, 
                    batch_size=12, 
                    max_length=1024,
                    )['dense_vecs']
        embeddings.append(embedding)
    return embeddings

def main():
    model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True)
    sentences = ["What is BGE?", "What is Amazon Bedrock?"]
    embeddings = embed_chunks(sentences, model)


if __name__ == "__main__":
    main()