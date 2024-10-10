'''
    File: generate_batch.py
    Description: This file generates the OpenAI batch json files for the zero-shot baseline tests.
'''

import json
from tqdm import tqdm

PROMPT = '''You will be provided 2 research paper’s title and abstract. Please determine which of the 2 articles is more novel. Follow these steps for evaluation.
            Step 1: Identity the problem and solution that the research paper attempts to solve.
            Step 2: Determine how unique the solution is given the current research landscape in 2024. Does the paper introduce a new idea, theory, or concept that has not been previously discussed in literature?
            Step 3: Determine how creative the solution is given the current research landscape in 2024. Does it apply a known idea in a completely new context or in a way that has not been done before?
            Step 4: Using the findings from steps 1-3, determine which paper is more novel.
            In your response please only state which paper is move novel. (e.g. 1 if paper 1 is more novel; 2 if paper 2 is more novel)'''

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
                            "content": f"Paper 1 Title: {papers['paper1']['title']}\nPaper 1 Abstract: {papers['paper1']['abstract']}\n\nPaper 2 Title: {papers['paper2']['title']}\nPaper 2 Abstract: {papers['paper2']['abstract']}"
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
                            "content": f"Paper 1 Title: {papers['paper2']['title']}\nPaper 1 Abstract: {papers['paper2']['abstract']}\n\nPaper 2 Title: {papers['paper1']['title']}\nPaper 2 Abstract: {papers['paper1']['abstract']}"
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