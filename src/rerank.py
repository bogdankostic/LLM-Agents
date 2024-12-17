import pathlib
import textwrap
import google.generativeai as genai
import jsonlines
from IPython.display import display
from IPython.display import Markdown
import random
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import pickle
from openai import OpenAI
import json
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import argparse

system_instruction = """
        You are a helpful assistant specializing in being able to understand scientific abstracts and matching social media comments to scientific papers.

        Specifically, your task is as follows:

        1. You will be given a Reddit post that references a certain paper published on Arxiv, but the post is lacking a citation or link to the paper.
        2. You will also be given a list of descriptions of papers on Arxiv. These papers have been curated as those that are most relevant to the contents of the post. The description of the papers will contain the id of the paper, the title of the paper, and the abstract.
        3. Your job is to identify what paper the Reddit post is referencing. Consider the claims made in the post and how these claims may match those in the paper. Output the id and title of the referenced paper.

        Use this JSON schema:

        Return {'id': str, 'title': str}
    """


def extract_paper_description(papers: list[dict], num_papers: int = 10):
    """Extracts the arXiv ID, title, and abstract from the first `num_papers` papers
    from a provided list of papers."""

    paper_descriptions: list[dict] = []
    for i in range(min(num_papers, len(papers))):
        paper = papers[i]
        description = {
            "id": paper["arxiv_id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
        }
        paper_descriptions.append(description)
    return paper_descriptions


def create_prompts(clean_reddit_posts: list[str], paper_descriptions: list[dict]):
    """Generates the prompt for each paper."""

    prompt = """
    Please help me identify the paper that this Reddit post most closely references.

    Reddit post: {clean_reddit_post}
    Relevant papers: {paper_description}"""
    filled_in_prompts = []
    for clean_reddit_post, paper_description in zip(
        clean_reddit_posts, paper_descriptions
    ):
        filled_in_prompt = prompt.format(
            clean_reddit_post=clean_reddit_post, paper_description=paper_description
        )
        filled_in_prompts.append(filled_in_prompt)
    return filled_in_prompts


def prepare_dataframe(rankings, comments, num_papers=10):
    """Prepare a dataframe for reranking using the rankings and comments dataframes"""
    rankings["paper_descriptions"] = rankings["ranked_docs"].apply(
        lambda papers: extract_paper_description(papers, num_papers)
    )
    rankings_joined_comments = rankings.join(comments, "comment_id")
    rankings_joined_comments["prompts"] = create_prompts(
        rankings_joined_comments["clean_reddit_post"],
        rankings_joined_comments["paper_descriptions"],
    )
    return rankings_joined_comments


def generate_openai_responses(
    prepared_dataframe, results_folder, client, model, unique_identifier=None
):

    responses_path = os.path.join(results_folder, f"{unique_identifier}_responses.pkl")

    # functionality to save intermediate progress
    if unique_identifier is not None and os.path.exists(responses_path):
        with open(responses_path, "rb") as f:
            responses = pickle.load(f)
    else:
        responses = {}

    for index, prompt in tqdm(
        zip(prepared_dataframe.index, prepared_dataframe.prompts),
        total=len(prepared_dataframe.index),
    ):

        if index in responses:
            continue

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
        )
        response = completion.choices[0].message
        responses[index] = response

        if unique_identifier is not None:
            with open(responses_path, "wb") as f:
                pickle.dump(responses, f)

    return responses


def generate_google_responses(
    prepared_dataframe, results_folder, model, unique_identifier=None
):
    gemini = genai.GenerativeModel(
        model_name=model, system_instruction=system_instruction
    )

    responses_path = os.path.join(results_folder, f"{unique_identifier}_responses.pkl")

    # functionality to save intermediate progress
    if unique_identifier is not None and os.path.exists(responses_path):
        with open(responses_path, "rb") as f:
            responses = pickle.load(f)
    else:
        responses = {}

    for index, prompt in tqdm(
        zip(prepared_dataframe.index, prepared_dataframe.prompts),
        total=len(prepared_dataframe.index),
    ):

        if index in responses:
            continue

        response = gemini.generate_content(prompt).text
        responses[index] = response

        if unique_identifier is not None:
            with open(responses_path, "wb") as f:
                pickle.dump(responses, f)

    return responses


def get_paper_descriptions(row):
    paper_descriptions = row["paper_descriptions"]
    if isinstance(paper_descriptions, str):
        paper_descriptions = eval(paper_descriptions)
    return paper_descriptions


def calculate_soft_retrieval_performance(row):
    if str(row["arxiv_id"]) in str(row["paper_descriptions"]):
        return 1
    else:
        return 0


def calculate_reranked_performance(row):
    if str(row["arxiv_id"]) in str(row["response"]):
        return 1
    else:
        return 0


def calculate_rank_chosen(row):
    for i, ranked_doc in enumerate(row["paper_descriptions"]):
        if str(ranked_doc["id"]) in row["response"]:
            return i
    return -1


def calculate_reciprocal_rank(row):

    arxiv_ids = [
        paper_description["id"] for paper_description in row["paper_descriptions"]
    ]
    for arxiv_id in arxiv_ids:
        if str(arxiv_id) in str(row["response"]):
            arxiv_ids.remove(arxiv_id)
            arxiv_ids.insert(0, arxiv_id)
            break

    # compute new rank with the correct one
    for i, arxiv_id in enumerate(arxiv_ids):
        if str(arxiv_id) == str(row["arxiv_id"]):
            return 1 / (i + 1)
    return 0


def calculate_performance(results_df):
    results_df["paper_descriptions"] = results_df.apply(get_paper_descriptions, axis=1)
    results_df["soft_retrieval_performance_top10"] = results_df.apply(
        calculate_soft_retrieval_performance, axis=1
    )
    results_df["reranked_performance"] = results_df.apply(
        calculate_reranked_performance, axis=1
    )
    results_df["calculate_rank_chosen_top10"] = results_df.apply(
        calculate_rank_chosen, axis=1
    )
    results_df["recall@1"] = results_df["reranked_performance"]
    results_df["reciprocal_rank_top10"] = results_df.apply(
        calculate_reciprocal_rank, axis=1
    )

    return results_df

def print_performance(results_df):
  print("mean recall@1", np.mean(results_df['recall@1']))
  print("std recall@1", np.std(results_df['recall@1']))
  print("se recall@1", stats.sem(results_df['recall@1']))

  print("mean reciprocal rank", np.mean(results_df['reciprocal_rank_top10']))
  print("std reciprocal rank", np.std(results_df['reciprocal_rank_top10']))
  print("se reciprocal rank", stats.sem(results_df['reciprocal_rank_top10']))

if __name__ == "__main__":

    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    llama_client = OpenAI(
        api_key=os.getenv("LLAMA_API_KEY"), base_url=os.getenv("LLAMA_URL")
    )
    llama_model = "llama3.1-70b-instruct-berkeley"

    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=str)
    parser.add_argument("--comments", type=str)
    parser.add_argument("--model", choices=["gemini", "gpt4omini", "gpt3.5", "llama"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--o", type=str)
    args = parser.parse_args()

    rankings_path = args.rankings
    comments_path = args.comments
    k = args.k
    model = args.model
    output_path = args.o
    assert (
        model == "gemini"
        or model == "gpt4omini"
        or model == "gpt3.5"
        or model == "llama"
    )

    print("Reading comments")
    comments = pd.read_csv(comments_path)
    comments.index = comments["id"]
    print(f"Read {len(comments)} comments")

    print("Reading retrieved paper rankings")
    with open(rankings_path, "r") as f:
        rankings = pd.DataFrame(json.load(f))
    print(f"Read {len(rankings)} rankings")

    length = min(len(comments), len(rankings))
    comments, rankings = comments.iloc[:length], rankings.iloc[:length]

    print("Preparing for LLM")
    prepared = prepare_dataframe(rankings, comments, k)

    os.makedirs("reranker_results", exist_ok=True)

    print("Querying LLM")
    if model == "gemini":
        responses = generate_google_responses(
            prepared, "reranker_results/", "gemini-1.5-flash"
        )
    elif model == "gpt4omini":
        responses = generate_openai_responses(
            prepared, "reranker_results/", openai_client, "gpt-4o-mini"
        )
    elif model == "gpt3.5":
        responses = generate_openai_responses(
            prepared, "reranker_results/", openai_client, "gpt-3.5-turbo"
        )
    else:
        responses = generate_openai_responses(
            prepared, "reranker_results/", llama_client, llama_model
        )

    print("Calculating performance")
    results = prepared.copy()
    results["response"] = responses
    results = calculate_performance(results)
    print_performance(results)

    print("Writing results")
    results.to_csv(output_path)
    print("Wrote results")
