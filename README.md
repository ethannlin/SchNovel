<div align="center">

# SchNovel

[![Paper](http://img.shields.io/badge/paper-arxiv.2409.16605-B31B1B.svg)](https://arxiv.org/abs/2409.16605)
[![Dataset](https://img.shields.io/badge/dataset-Hugging_Face-yellow)](https://huggingface.co/datasets/ethannlin/SchNovel/tree/main)

</div>

## Description

The repository contains the code for the paper _"Evaluating and Enhancing Large Language Models for Novelty Assessment in Scholarly Publications."_ Each folder contains the basic file structure and code required to replicate the experiments and the SchNovel benchmark proposed in the paper.

## Installation

### Using Pip

1. Clone the project repository:

    ```bash
    git clone https://github.com/ethannlin/schnovel
    cd schnovel
    ```

2. [Optional] Create and activate a conda environment:

    ```bash
    conda create -n myenv python=3.12
    conda activate myenv
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Using Conda

1. Clone the project repository:

    ```bash
    git clone https://github.com/ethannlin/schnovel
    cd schnovel
    ```

2. Create a conda environment and install dependencies:

    ```bash
    conda env create -f environment.yaml -n myenv
    ```

3. Activate the conda environment:

    ```bash
    conda activate myenv
    ```

## Setting up

### Download the labeled datasets

```bash
git clone https://huggingface.co/datasets/ethannlin/SchNovel
```

### Setup .env file

1. Update the `.env` file to include your OpenAI API keys:

    ```bash
    OPENAI_API_KEY = ""
    OPENAI_ORG_ID = ""
    OPENAI_PROJECT_ID = ""
    ```

2. Replace the empty strings (`""`) with your actual API key, organization ID, and project ID.

### Setup RAG-Novelty

1. Navigate to `rag-novelty` folder.

    ```bash
    cd rag-novelty
    ```

2. Update `scripts/generate.py` with the filepaths to the vector db data, the desired directory path, and database name.
    - Running this script will create a vector database for the desired category.
    ```bash
    python scripts/generate.py
    ```

## How to run

1. Navigate to the project folder:

    ```bash
    cd [$folder_name]
    ```

2. Update `generate_batch.py`, `average_results.ipynb`, and `que_batch.ipynb` with the desired category and file paths.

    ```bash
    # example: replace with filepath to [CATEGORY]'s json dataset
    FILEPATH = ""
    ```

3. Run `generate_batch.py`

    - This will generate and store the batch files into the desired directory.

    ```bash
    python generate_batch.py
    ```

4. Open `que_batch.ipynb`

    - Follow instructions in the notebook to queue the batches either manually or automatically.

5. Retrieve results from [OpenAI Batch API](https://platform.openai.com/batches) and run `average_results.ipynb`.
