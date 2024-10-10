'''
    File: generate_batch.py
    Description: This file generates the OpenAI batch json files for RAG-Novelty tests.
'''

from scripts.chroma_utils import get_vector_store
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import json
from tqdm import tqdm
from datetime import datetime

# Name of the collection (vector db)
COLLECTION_NAME = ""

# Directory path to [CATEGORY] vector db
DB_DIR = ""

# Filepath to [CATEGORY] dataset
FILEPATH = ""

# Directory to store created batch files
BATCH_DIR = ""

# Function to get relevant context from the vector store
def get_timestamp_for_paper(paper, date, k=10):
    query_embedding = embedding_model.embed_query(paper['abstract'])

    # Prepare filter condition based on paper's timestamp ($lt = less than)
    filter_condition = {
        "unix_timestamp": {"$lt": int(date)}
    }

    results = client.similarity_search_by_vector_with_relevance_scores(
        query_embedding,
        k=k,
        filter=filter_condition
    )

    avg_timestamp = []
    avg_relevance = []

    for result in results:
        metadata = result[0].metadata
        avg_timestamp.append(int(metadata["unix_timestamp"]))
        avg_relevance.append(result[1])

    avg_timestamp = sum(avg_timestamp) / len(avg_timestamp)
    avg_relevance = sum(avg_relevance) / len(avg_relevance)

    return datetime.fromtimestamp(avg_timestamp), avg_relevance

PROMPT = '''You are an advanced language model tasked with determining the novelty of research papers in 2024. Your goal is to evaluate and compare the novelty of two research papers based on their titles and abstracts. The order in which the papers are presented is random and should not influence your evaluation.

Step 1: Independent Evaluation

Analyze each research paper’s title and abstract independently. Treat each paper as if it is the only one under review at that moment.
Retrieve similar abstracts from a vector database based on the provided abstracts.
Contextual Date Analysis: Average the published dates of the retrieved documents. Use this average date as additional context for your evaluation.
Consider that papers with an average date that is later or more recent in time are generally more novel.
Consider the following aspects for each paper:
Novelty of Methodology: Are the methods used new and innovative?
Surprisingness of Findings: Are the findings unexpected or counterintuitive?
Impact on Existing Knowledge: How does the research challenge or expand current scientific understanding?
Potential for Future Research: Does the paper open up new directions for research?
Relevance to 2024 Scientific Understanding: How well does the paper align with or push the boundaries of current trends?
Step 2: Quantitative Assessment

Assign a score from 1-10 to each research paper for its novelty, with 10 being the most novel. This score should be based on the content of the title and abstract, as well as the contextual information from the average published dates.
Provide a brief justification for the score, using specific quotes and context.
Step 3: Final Comparison

After independently scoring each paper, compare the scores.
Determine which paper exhibits greater novelty based on the higher score, and conclude with: "The more novel and impactful paper is [Paper X or Paper Y].
Important: The order of presentation is random and should not influence your decision. Evaluate each paper strictly on its content and merit, incorporating the additional context from the vector database as described.'''

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
client = get_vector_store(COLLECTION_NAME, DB_DIR)

with open(FILEPATH, "r") as file:
    data = json.load(file)

data = data['data']

# Create batch files
openai_client = OpenAI()

for year, gap in data.items():
    for gap, papers in gap.items():
        batch = []
        counter = 1
        for paper in tqdm(papers):
            paper1_title = paper['paper1']['title']
            paper2_title = paper['paper2']['title']
            paper1_abstract = paper['paper1']['abstract']
            paper2_abstract = paper['paper2']['abstract']

            if paper['paper1']['unix_timestamp'] > paper['paper2']['unix_timestamp']:
                date = paper['paper1']['unix_timestamp']
            else:
                date = paper['paper2']['unix_timestamp']

            context1 = get_timestamp_for_paper(paper['paper1'], date)
            context2 = get_timestamp_for_paper(paper['paper2'], date)

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
                                f"Paper X Average Cosine Similarity (higher is better): {context1[1]}\n"
                                f"Paper X Average Contextual Date (more recent is better): {context1[0]}\n"
                                f"Paper Y Average Cosine Similarity (higher is better): {context2[1]}\n"
                                f"Paper Y Average Contextual Date (more recent is better): {context2[0]}\n"
                                f"Paper X Title: {paper1_title}\n"
                                f"Paper X Abstract: {paper1_abstract}\n"
                                f"Paper Y Title: {paper2_title}\n"
                                f"Paper Y Abstract: {paper2_abstract}"
                            )
                        }
                    ]
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
                                f"Paper X Average Cosine Similarity (higher is better): {context2[1]}\n"
                                f"Paper X Average Contextual Date (more recent is better): {context2[0]}\n"
                                f"Paper Y Average Cosine Similarity (higher is better): {context1[1]}\n"
                                f"Paper Y Average Contextual Date (more recent is better): {context1[0]}\n"
                                f"Paper X Title: {paper2_title}\n"
                                f"Paper X Abstract: {paper2_abstract}\n"
                                f"Paper Y Title: {paper1_title}\n"
                                f"Paper Y Abstract: {paper1_abstract}"
                            )
                        }
                    ]
                }
            }

            batch.append(input)
            counter += 1

        # Write batch to after every gap
        with open(f"{BATCH_DIR}/{year}_{gap}_batch.json", "w") as file:
            for item in batch:
                json.dump(item, file)
                file.write("\n")