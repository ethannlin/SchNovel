'''
    File: r1prof_batch.py
    Description: This file generates the OpenAI batch json files for round 1 discussions for the professor role in LLM Discussion tests.
'''

import json
from tqdm import tqdm

CATEGORY = ""

PROMPT = f'''You are a professor specializing in all areas of {CATEGORY}. You will be provided with the titles and abstracts of two research papers. Your task is to determine which of the two articles is more novel by evaluating their originality, contribution to the field, and potential impact. Focus on aspects such as new methodologies, unexplored problems, innovative solutions, and how the work advances the state of the art.
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

# Path to results from "editor-role"
EDITOR_RESULTS = ""

# Path to results from "phd-role"
PHD_RESULTS = ""

# Path to results from "prof-role"
PROF_RESULTS = ""


with open(FILEPATH, "r") as file:
    data = json.load(file)

data = data['data']

for year, gap in data.items():
    for gap, papers in gap.items():
        batch = []
        counter = 1

        prof_responses = []
        with open(f"{PROF_RESULTS}/{year}_{gap}_results.jsonl", "r") as f:
            for line in f:
                prof_responses.append(json.loads(line))

        phd_responses = []
        with open(f"{PHD_RESULTS}/{year}_{gap}_results.jsonl", "r") as f:
            for line in f:
                phd_responses.append(json.loads(line))

        editor_responses = []
        with open(f"{EDITOR_RESULTS}/{year}_{gap}_results.jsonl", "r") as f:
            for line in f:
                editor_responses.append(json.loads(line))

        for i, paper in tqdm(enumerate(papers)):
            paper1_title = paper['paper1']['title']
            paper2_title = paper['paper2']['title']
            paper1_abstract = paper['paper1']['abstract']
            paper2_abstract = paper['paper2']['abstract']

            prof_response = prof_responses[i*2]['response']['body']['choices'][0]['message']['content']
            phd_response = phd_responses[i*2]['response']['body']['choices'][0]['message']['content']
            editor_response = editor_responses[i*2]['response']['body']['choices'][0]['message']['content']

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
                        },
                        {
                            "role": "assistant",
                            "content": f"{prof_response}"
                        },
                        {
                            "role": "user",
                            "content": (
                                "These are responses from another reviews. Please revise your response if necessary and still limit your response to 150 tokens max.\n"
                                f"Response 1: {phd_response}\n"
                                f"Response 2: {editor_response}"
                            )
                        }
                    ],
                    "max_completion_tokens": 200
                }
            }

            batch.append(input)
            counter += 1

            prof_response = prof_responses[i*2 + 1]['response']['body']['choices'][0]['message']['content']
            phd_response = phd_responses[i*2 + 1]['response']['body']['choices'][0]['message']['content']
            editor_response = editor_responses[i*2 + 1]['response']['body']['choices'][0]['message']['content']

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
                        {
                            "role": "assistant",
                            "content": f"{prof_response}"
                        },
                        {
                            "role": "user",
                            "content": (
                                "These are responses from another reviews. Please revise your response if necessary and still limit your response to 150 tokens max.\n"
                                f"Response 1: {phd_response}\n"
                                f"Response 2: {editor_response}"
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