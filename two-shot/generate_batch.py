'''
    File: generate_batch.py
    Description: This file generates the OpenAI batch json files for the two-shot baseline tests.
'''

import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime

PROMPT = '''You will be provided 2 research paper’s title and abstract. Please determine which of the 2 articles is more novel. Follow these steps for evaluation.
            Step 1: Identity the problem and solution that the research paper attempts to solve.
            Step 2: Determine how unique the solution is given the current research landscape in 2024. Does the paper introduce a new idea, theory, or concept that has not been previously discussed in literature?
            Step 3: Determine how creative the solution is given the current research landscape in 2024. Does it apply a known idea in a completely new context or in a way that has not been done before?
            Step 4: Using the findings from steps 1-3, determine which paper is more novel.
            In your response please only state which paper is move novel. (e.g. 1 if paper 1 is more novel; 2 if paper 2 is more novel)'''

# Replace with filepath to [CATEGORY]'s json dataset
FILEPATH = ""

# Replace with filepath to [CATEGORY]'s full unsampled dataset
SAMPLE_DIR = ""

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

    It also randomly samples two example pairs from the updated arXiv collection (does not contain papers that would've already been sampled),
    and includes it in the batch file prompts for reference.
'''
# one year in seconds
one_year = 365 * 24 * 60 * 60

df = pd.read_json(SAMPLE_DIR, lines=True)

for year, gap in data.items():
    # Filter out papers for the desired year
    unix_year = datetime(int(year), 1, 1).timestamp()
    df1 = df[(df['created'] >= unix_year) & (df['created'] < unix_year + one_year)]

    for gap, papers in gap.items():
        batch = []  # Variable to store batch inputs
        counter = 1 # Variable to increment batch input IDs

        # Filter out papers for the desire year gap
        unix_year_gap = datetime(int(year) - int(gap), 1, 1).timestamp()
        df2 = df[(df['created'] >= unix_year_gap) & (df['created'] < unix_year_gap + one_year)]

        for i, paper in tqdm(enumerate(papers)):
            paper1_title = paper['paper1']['title']
            paper2_title = paper['paper2']['title']
            paper1_abstract = paper['paper1']['abstract']
            paper2_abstract = paper['paper2']['abstract']

            # Sample two paper pairs for two-shot
            paper3 = df1.sample(1)
            df1 = df1.drop(paper3.index)
            paper4 = df2.sample(1)
            df2 = df2.drop(paper4.index)

            paper5 = df1.sample(1)
            df1 = df1.drop(paper5.index)
            paper6 = df2.sample(1)
            df2 = df2.drop(paper6.index)

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
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper 1 Title: {paper3['title'].values[0]}\n"
                                f"Paper 1 Abstract: {paper3['abstract'].values[0]}\n"
                                f"Paper 2 Title: {paper4['title'].values[0]}\n"
                                f"Paper 2 Abstract: {paper4['abstract'].values[0]}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: 1"
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper 1 Title: {paper6['title'].values[0]}\n"
                                f"Paper 1 Abstract: {paper6['abstract'].values[0]}\n"
                                f"Paper 2 Title: {paper5['title'].values[0]}\n"
                                f"Paper 2 Abstract: {paper5['abstract'].values[0]}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: 2"
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper 1 Title: {paper1_title}\n"
                                f"Paper 1 Abstract: {paper1_abstract}\n"
                                f"Paper 2 Title: {paper2_title}\n"
                                f"Paper 2 Abstract: {paper2_abstract}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A:"
                        }
                    ]
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
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper 1 Title: {paper3['title'].values[0]}\n"
                                f"Paper 1 Abstract: {paper3['abstract'].values[0]}\n"
                                f"Paper 2 Title: {paper4['title'].values[0]}\n"
                                f"Paper 2 Abstract: {paper4['abstract'].values[0]}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: 1"
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper 1 Title: {paper6['title'].values[0]}\n"
                                f"Paper 1 Abstract: {paper6['abstract'].values[0]}\n"
                                f"Paper 2 Title: {paper5['title'].values[0]}\n"
                                f"Paper 2 Abstract: {paper5['abstract'].values[0]}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: 2"
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper 1 Title: {paper2_title}\n"
                                f"Paper 1 Abstract: {paper2_abstract}\n"
                                f"Paper 2 Title: {paper1_title}\n"
                                f"Paper 2 Abstract: {paper1_abstract}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A:"
                        }
                    ]
                }
            }

            batch.append(input)
            counter += 1

        # Write batch to after every gap
        with open(f"{BATCH_DST}/{year}_{gap}_batch.json", "w") as file:
            for item in batch:
                json.dump(item, file)
                file.write("\n")