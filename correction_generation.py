import json
import pandas as pd
import os
from openai import OpenAI
import time
from utils import preprocess_dataframe, prepare_context_prompt

# ADD CONTEXT PROMPT TO CSV
# ADD ANSWERS TO CSV
# ADD CONTEXT 

API_KEY = os.getenv('OPENAI_API_KEY')

os.makedirs('data/processed_context', exist_ok=True)

def gpt(api_key: str, prompt: str) -> str | None:
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    max_retries = 20
    curr_tries = 1
    while curr_tries <= max_retries:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                timeout=10
            )
            message = response.choices[0].message.content
            return message
        except Exception as e:
            print('\t'+str(e))
            curr_tries += 1
            time.sleep(5)
            continue
    return ''

def clean_processed_prompt(prompt: str) -> str:
    return prompt.split("Given a text, you are required to generate three different queries from the text for")[0]


if __name__ == "__main__":
    data = pd.read_csv('data/online_posts/Politifact_data.csv')[20000:22000].reset_index(drop=True)
    processed_df = preprocess_dataframe(data, 'Post', 'StartDate')
    rows = processed_df.iloc[[730, 165, 389, 1093, 720]].copy()

    data = json.load(open('data/processed/selected_posts_with_responses.json', 'r'))


    for idx in data:
        context_prompt = prepare_context_prompt(
            text=clean_processed_prompt(data[idx]['processed']),
            relevant_context=data[idx]['post_context']
        )
        print(context_prompt)

    
    
