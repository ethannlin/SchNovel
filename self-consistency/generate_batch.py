'''
    File: generate_batch.py
    Description: This file generates the OpenAI batch json files for the self-consistency baseline tests.
'''

import json
from tqdm import tqdm

PROMPT = '''You will be provided 2 research paper’s title and abstract. Please determine which of the 2 articles is more novel.
            Please limit your response to 150 tokens max. In your response please conclude with: "The more novel and impactful paper is [Paper X or Paper Y]"'''

# Replace with filepath to [CATEGORY]'s json dataset
FILEPATH = ""

# Replace with destination filepath for batch files
BATCH_DST = ""

with open(FILEPATH, "r") as file:
    data = json.load(file)

data = data['data']

for year, gap in data.items():
    for gap, papers in gap.items():
        batch = []
        counter = 1
        for paper in tqdm(papers):
            paper1_title = paper['paper1']['title']
            paper2_title = paper['paper2']['title']
            paper1_abstract = paper['paper1']['abstract']
            paper2_abstract = paper['paper2']['abstract']

            input = {
                "custom_id": f"{year}_{gap}_{counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": PROMPT
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Paper X Title: {paper1_title}\n"
                                f"Paper X Abstract: {paper1_abstract}\n"
                                f"Paper Y Title: {paper2_title}\n"
                                f"Paper Y Abstract: {paper2_abstract}"
                            )
                        }
                    ],
                    "max_completion_tokens" : 200,
                    "n": 10,
                    "temperature": 0.7
                }
            }

            batch.append(input)
            counter += 1

            input = {
                "custom_id": f"{year}_{gap}_{counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": PROMPT
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Paper X Title: {paper2_title}\n"
                                f"Paper X Abstract: {paper2_abstract}\n"
                                f"Paper Y Title: {paper1_title}\n"
                                f"Paper Y Abstract: {paper1_abstract}"
                            )
                        },
                    ],
                    "max_completion_tokens" : 200,
                    "n": 10,
                    "temperature": 0.7
                }
            }

            batch.append(input)
            counter += 1

        # Write batch to after every gap
        with open(f"{BATCH_DST}/{year}_{gap}_batch.json", "w") as file:
            for item in batch:
                json.dump(item, file)
                file.write("\n")