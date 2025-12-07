import pandas as pd
from utils import preprocess_dataframe
import os
from openai import OpenAI
import time
import json

API_KEY = os.getenv('OPENAI_API_KEY')

os.makedirs('data/processed', exist_ok=True)

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

def generate_queries(row: pd.Series) -> pd.Series:
    if API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    prompt = row['processed']
    response = gpt(API_KEY, prompt)

    if response:
        responses = response.split('\n')
        for r in responses:
            if r.startswith('1.'):
                row['response_1'] = r[2:].strip()
            elif r.startswith('2.'):
                row['response_2'] = r[2:].strip()
            elif r.startswith('3.'):
                row['response_3'] = r[2:].strip()
    return row

if __name__ == "__main__":
    data = pd.read_csv('data/online_posts/Politifact_data.csv')[20000:22000].reset_index(drop=True)
    processed_df = preprocess_dataframe(data, 'Post', 'StartDate')
    rows = processed_df.iloc[[730, 165, 389, 1093, 720]].copy()
    
    answer_map = json.load(open('data/processed/selected_posts_with_responses.json', 'r'))

    for idx, m in answer_map.items():
        idx = int(idx)
        rows.loc[idx, "response_1"] = m["1st_response"]
        rows.loc[idx, "response_2"] = m["2nd_response"]
        rows.loc[idx, "response_3"] = m["3rd_response"]
        rows.loc[idx, "augmented_prompt"] = m["augmented"]
        rows.loc[idx, "response_1_augmented"] = m["1st_response_augmented"]
        rows.loc[idx, "response_2_augmented"] = m["2nd_response_augmented"]
        rows.loc[idx, "response_3_augmented"] = m["3rd_response_augmented"]
        rows.loc[idx, "processed_augmented"] = m["processed_prompt_augmented"]
    
    rows.to_csv('data/processed/selected_posts_with_responses_test.csv', index=False)
    
    