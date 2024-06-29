import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


keyword = 'israel news'
api_url = 'https://api.hasdata.com/scrape/google'

headers = {'x-api-key': os.environ.get('HASDATA_API_KEY')}

params = {
    'q': keyword,
    'domain': 'google.com',
    'tbm': 'nws',
    'num': 1000
}

response = requests.get(api_url, params=params, headers=headers)

data = response.json()
news = data['newsResults']
df = pd.DataFrame(news)
df["tag"] = keyword
df.to_json(f"../data/{keyword.replace(' ', '_')}.json", orient='records')
