# CS194/294-196: Large Language Model Agents
## Project: Towards Recovering Citations Missing from Social Media Posts: A Benchmark for the Retrieval of Primary Sources Implicitly Referenced in Reddit Comments
Team Name: The Scientists  
Track: Benchmarks  
Team Members: Bogdan Kostić, Yantong Zhong, Dan Hickey, Kanav Mittal, Lydia Wang

### 1) Installation
To install the required packages, run the following command:
```bash
git clone https://github.com/bogdankostic/reddit-citation-recovery.git
cd reddit-citation-recovery
pip install -r requirements.txt
```

### 2) Set up Vector DB
To set up the Vector DB, run the following command:
```bash
docker compose -f weaviate/docker-compose.yml up -d
```

### 3) Index Paper Corpus
To index the [Dataclysm paper](somewheresystems/dataclysm-arxiv) corpus in the Vector DB, run the following command:
```bash
python src/index_articles.py
```

### 4) Rewrite Reddit Comments
To rewrite the Reddit comments using LLMs, run the following command:
```bash
python src/rewrite.py \
  --file_path <PATH_TO_REDDIT_COMMENTS_CSV> \
  --column_names <COLUMN_NAMES_TO_REWRITE> \
  --llm <ONE OF "gemini-1.5-flash", "gpt-4o-mini", "gpt-3.5-turbo", OR "llama3.1-70b-instruct-berkeley"> \
  --prompt <ONE OF "simulate_abstract", "remove_text", "scientific_style", "search_query"> \
```

### 5) Embed Reddit Comments
To embed the Reddit comments using the [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) model, run the following command:
```bash
python src/embed_comments.py \
  --file_path <PATH_TO_REDDIT_COMMENTS_CSV> \
  --column_names <COLUMN_NAMES_TO_EMBED>
```

### 6) Retrieve Papers and Evaluate Results
To retrieve papers for the Reddit comments using both BM25 and dense embedding retrieval, and evaluate the results, run the following command:
```bash
python src/assess_performance.py \
  --file_path <PATH_TO_REDDIT_COMMENTS_CSV_WITH_EMBEDDINGS> \
  --column_name <COLUMN_NAME_TO_USE_FOR_BM25_EVALUATION> \
  --embedding_column <COLUMN_CONTAINING_EMBEDDINGS_FOR_DENSE_RETRIEVAL> 
```

### 7) Rerank Retrieved Papers
To rerank the retrieved papers using LLMs and evaluate the results, run the following command:
```bash
python src/rerank.py \
  --rankings <PATH_TO_RANKINGS_JSON_OUTPUT_BY_STEP6> \
  --comments <PATH_TO_CSV_WITH_SAMPLE_COMMENTS> \
  --model <ONE OF "gemini", "gpt4omini", "gpt3.5", OR "llama"> \
  --k <K TOP PAPERS ARE INCLUDED IN PROMPT, DEFAULT 10> \
  --o <OUTPUT CSV FOR RESULTS>
```
