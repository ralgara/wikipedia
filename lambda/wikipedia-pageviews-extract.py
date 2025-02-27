#!/usr/bin/env python3
import json
import os

with open(os.path.expanduser('~/Desktop/pageviews_20250101.json')) as f:
    s = json.load(f)
    s = s['items'][0]
    date = f"{s['year']}-{s['month']}-{s['day']}"
    [ {**z, 'date':date} for z in s['articles'] ]
print(s)

# {'project': 'en.wikipedia', 'access': 'all-access', 'year': '2025', 'month': '01', 'day': '01', 'articles': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...]}