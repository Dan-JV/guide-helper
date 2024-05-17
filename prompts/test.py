
from jinja2 import Environment, FileSystemLoader, select_autoescape

# load in the guide_helper.jinja file

def create_template(template_file: str):
    """Create a Jinja template from a template file.
    Example:
    ```
    create_template("prompts/guide_helper.jinja")
    ```
    """
    jinja_env = Environment(
        loader=FileSystemLoader(searchpath="./"), autoescape=select_autoescape()
    )
    template = jinja_env.get_template(template_file)
    return template


template = create_template("prompts/guide_helper.jinja")
print(template)

from jinja2 import Template

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = Template(
    "Some text is provided below. Given the text, extract up to {{ max_keywords }} "
    "keywords from the text. Avoid stopwords.\n"
    "---------------------\n"
    "{{ text }}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
print(DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL)

from langchain_core.prompts import PromptTemplate

template2 = PromptTemplate(template=template)

template3 = PromptTemplate.from_template(DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, template_format="jinja2")

print()