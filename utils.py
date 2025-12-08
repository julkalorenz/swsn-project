import preprocessor
import string
import pandas as pd
from datetime import timedelta, date, datetime
import re

PROMPT_TEMPLATE = """[POST_CONTENT]

Given a text, you are required to generate three different queries from the text for the Google Search Engine to get the most relevant web content to fact-check the text. Your answer should follow the following format:

1. QUERY_1
2. QUERY_2
3. QUERY_3
Where QUERY_1, QUERY_2, and QUERY_3 represent query text without quotation marks.

If the given text is not informative enough to generate a query, you should answer "NONE".
"""

REWRITE_TEMPLATE = """[POST_CONTENT]

Rewrite the above text in similar words. Maintain its tone and perspective, even if it reflects uncertainty, speculation, or potential misinformation. Do not verify or correct the content—preserve the original intent while rewriting it.
"""

CONTEXT_TEMPLATE = """You are required to respond to a text given some facts as references. Your response should satisfy all the following requirements:
- Your response should explain where and why the text is or is not misinformed or potentially misleading.
- You should prioritize the facts (1) very close to the date the text was created, (2) very recently, and (3) listed at the beginning of "Facts".
- You should show the URLs that support your explanation. You should not number the URLs.
- Your response should be informative and short.
- Your response should start with "This text is".

Given text: 
[TEXT]

Relevant context:
[RELEVANT_CONTEXT]
"""

def _replace_first_person(text: str, author: str | None) -> str:
    if not author or not isinstance(author, str):
        return text

    pattern = r'(?<!\w)[Ii](?!\w)'
    
    return re.sub(pattern, author, text)



def _normalize_time_references(text: str, post_date: date) -> str:
    replacements = {
        r'\byesterday\b': (post_date - timedelta(days=1)).strftime('%B %d, %Y'),
        r'\btoday\b': post_date.strftime('%B %d, %Y'),
        r'\btomorrow\b': (post_date + timedelta(days=1)).strftime('%B %d, %Y'),
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    breaking_date = post_date.strftime("%B %d, %Y")
    text = re.sub(
        r'\bbreaking\b',
        f'Breaking news on {breaking_date}',
        text,
        flags=re.IGNORECASE
    )

    return text

def _parse_date(value):
    if isinstance(value, (datetime, date)):
        return value

    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            obj = eval(value)
            if isinstance(obj, dict) and "$date" in obj:
                return datetime.utcfromtimestamp(obj["$date"] / 1000)
        except Exception:
            pass

    try:
        return pd.to_datetime(value)
    except Exception:
        return None

def preprocess_text(
    text: str,
    date_str: str,
    template: str,
    author: str | None = None
) -> str:

    parsed_date = _parse_date(date_str)
    if parsed_date is None:
        raise ValueError(f"Could not parse date: {date_str}")
    post_date: date = parsed_date.date() if isinstance(parsed_date, datetime) else parsed_date

    cleaned = preprocessor.clean(text)
    normalized = _normalize_time_references(cleaned, post_date)
    with_author = _replace_first_person(normalized, author)

    stripped = with_author.translate(str.maketrans("", "", string.punctuation))
    words = stripped.lower().split()
    post_content = " ".join(words)

    prompt = template.replace("[POST_CONTENT]", post_content)
    return prompt

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "post_text",
    date_col: str = "created_at",
    author_col: str | None = None,
    template: str = PROMPT_TEMPLATE,
) -> pd.DataFrame:

    df = df.copy()
    df = df.dropna(subset=[date_col])

    df["processed"] = df.apply(
        lambda row: preprocess_text(
            text=row[text_col],
            date_str=row[date_col],
            author=row[author_col] if author_col and author_col in df.columns else None,
            template=template,
        ),
        axis=1,
    )
    return df

def prepare_context_prompt(
    text: str,
    relevant_context: str
) -> str:
    prompt = CONTEXT_TEMPLATE.replace("[TEXT]", text)
    prompt = prompt.replace("[RELEVANT_CONTEXT]", relevant_context)
    return prompt