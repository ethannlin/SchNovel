'''
    File: editor_batch.py
    Description: This file generates the OpenAI batch json files for the editor role in LLM Discussion tests.
'''

import json
from tqdm import tqdm

PROMPT = '''You are an editor for a prestigious journal specializing in all areas of computer science. You will be provided with the titles and abstracts of two research papers. Your task is to determine which of the two articles is more novel by evaluating their originality, contribution to the field, and potential for publication in a leading journal. Focus on aspects such as new methodologies, unexplored problems, innovative solutions, and how the work advances the state of the art in computer science.
Follow these steps for evaluation.
Step 1: Identity the problem and solution that the research paper attempts to solve.
Step 2: Determine how unique the solution is given the current research landscape in 2024. Does the paper introduce a new idea, theory, or concept that has not been previously discussed in literature?
Step 3: Determine how creative the solution is given the current research landscape in 2024. Does it apply a known idea in a completely new context or in a way that has not been done before?
Step 4: Using the findings from steps 1-3, determine which paper is more novel.
Please limit your response to 150 tokens max. In your response please conclude with: "The more novel and impactful paper is [Paper X or Paper Y]'''

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
                    "max_completion_tokens": 200
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
                        }
                    ],
                    "max_completion_tokens": 200
                }
            }

            batch.append(input)
            counter += 1

        # Write batch to after every gap
        with open(f"{BATCH_DST}/{year}_{gap}_batch.json", "w") as file:
            for item in batch:
                json.dump(item, file)
                file.write("\n")