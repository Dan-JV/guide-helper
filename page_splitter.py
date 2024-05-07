import re

def load_data():
    with open("data/support_articles_raw.txt", encoding="utf-8") as f:
        data = f.read()
    
    return data


data = load_data()


def split_html_pages(data):
    # Read the entire content of the original file

    # Split the content by the delimiter "----------"
    pages = data.split('----------')

    # Save each page into a separate file
    try:
        for i, page in enumerate(pages):
            if page.strip():  # Ensure the page is not just whitespace
                output_file_path = f'data/pages/page_{i + 1}.txt'
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(page.strip())

        print("split sucessfull")
    except:
        print("failed to split")

split_html_pages(data)