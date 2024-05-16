import polars as pl


def load_data(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as f:
        data = f.read()
    
    return data


def extract(data_file_path, metadata_file_path):
    data = load_data(data_file_path)
    metadata = pl.read_excel(metadata_file_path)

    return data, metadata