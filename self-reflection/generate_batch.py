'''
    File: generate_batch.py
    Description: This file generates the OpenAI batch json files for the self-reflection baseline tests.
'''

import json
from tqdm import tqdm

PROMPT = '''You are an advanced language model tasked with determining the novelty of research papers in 2024. Your goal is to evaluate and compare the novelty of two research papers based on their titles and abstracts. **The order in which the papers are presented is random and should not influence your evaluation.**

**Step 1: Independent Evaluation**

- Analyze each research paper’s title and abstract **independently**. Treat each paper as if it is the only one under review at that moment.
- Consider the following aspects for each paper:
    - **Novelty of Methodology**: Are the methods used new and innovative?
    - **Surprisingness of Findings**: Are the findings unexpected or counterintuitive?
    - **Impact on Existing Knowledge**: How does the research challenge or expand current scientific understanding?
    - **Potential for Future Research**: Does the paper open up new directions for research?
    - **Relevance to 2024 Scientific Understanding**: How well does the paper align with or push the boundaries of current trends?

**Step 2: Quantitative Assessment**

- Assign a score from 1-10 to each research paper for its novelty, with 10 being the most novel. This score should be based solely on the content of the title and abstract.
- Provide a brief justification for the score, using specific quotes and context.

**Step 3: Final Comparison**

- After independently scoring each paper, compare the scores.
- Determine which paper exhibits greater novelty based on the higher score, and provide the identifier (X or Y) of the more novel paper.

**Important**: The order of presentation is random and should not influence your decision. Evaluate each paper strictly on its content and merit.'''

# Replace with filepath to [CATEGORY] json dataset
FILEPATH = ""

# Replace with destination filepath for batch files
BATCH_DST = ""

with open(FILEPATH, "r") as file:
    data = json.load(file)

data = data['data']

'''
    Creating the batch files

    The following code loops through the dataset and creates a batch file for each individual year and year gap.
    The code also swaps the position of the two papers. Using the provided dataset, there will be 25 batch files
    saved to your local directory each with 100 pairs (200 including swapped pairs)
'''
for year, gap in data.items():
    for gap, papers in gap.items():
        batch = []
        counter = 1
        for i, paper in tqdm(enumerate(papers)):
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
                            "content": f"For the sake of time, please thoroughly evaluate, but only state the identifier (X or Y) of the research paper you believe is more novel. For example if Research Paper X is better, respond with X and vice versa. No other explanation is needed.\nResearch Paper X Title: {papers['paper1']['title']}\nResearch Paper X Abstract: {papers['paper1']['abstract']}\n\nResearch Paper Y Title: {papers['paper2']['title']}\nResearch Paper Y Abstract: {papers['paper2']['abstract']}"
                        },
                    ],
                }
            }

            batch.append(input)
            counter += 1

            # Swap positions of input papers in the prompt
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
                            "content": f"For the sake of time, please thoroughly evaluate, but only state the identifier (X or Y) of the research paper you believe is more novel. For example if Research Paper X is better, respond with X and vice versa. No other explanation is needed.\nResearch Paper X Title: {papers['paper2']['title']}\nResearch Paper X Abstract: {papers['paper2']['abstract']}\n\nResearch Paper Y Title: {papers['paper1']['title']}\nResearch Paper Y Abstract: {papers['paper1']['abstract']}"
                        },
                    ],
                }
            }

            batch.append(input)
            counter += 1

        # Write batch to after every gap (Modify the filepath if desired)
        with open(f"{BATCH_DST}/{year}_{gap}_batch.json", "w") as file:
            for item in batch:
                json.dump(item, file)
                file.write("\n")