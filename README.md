# PDF Search and QA with OpenAI and QDrant

This repository contains a Python script that extracts text from a PDF document, processes it into chunks, generates embeddings using OpenAI's ADA model, indexes the chunks in a QDrant vector database, and then performs question-answering using OpenAI GPT-3.5.

## Prerequisites

Before you begin, ensure you have the following:

1. **OpenAI API Key**: Obtain an API key from OpenAI by following their [API documentation](https://beta.openai.com/docs/). Set up the key as an environment variable `OPENAI_API_KEY`.

2. **QDrant Cloud API Key**: Sign up for an account on [QDrant Cloud](https://qdrant.cloud/) and obtain your API key.

3. **Python**: Install Python (>= 3.6).

## Setup

1. **Install the required Python packages using pip:**:

    pip install -r requirements.txt

2. **API Keys**

Open .env and replace OpenAI API key, QDrant URL, and QDrant API key with the one you got.

## Running the code

1. Place your PDF document in the same directory as main.py or provide its path in the `pdf_path`` variable in main.py.

2. **Run the script:**

    python main.py

3. Follow the prompts to interact with the code:

- Enter a question.

- The code will search for related chunks in the PDF and generate answers using GPT-3.5.

4. Repeat step 3 as desired