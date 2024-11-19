from ast import literal_eval

from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateBM25Retriever, WeaviateEmbeddingRetriever
import pandas as pd
from tqdm import tqdm


def recall(retrieved_docs, relevant_docs, top_k=None):
    recall_count = 0
    for cur_retrieved_docs, cur_relevant_doc in zip(retrieved_docs, relevant_docs):
        if top_k:
            cur_retrieved_docs = cur_retrieved_docs[:top_k]
        if cur_relevant_doc in cur_retrieved_docs:
            recall_count += 1

    return recall_count / len(relevant_docs)


def mean_reciprocal_rank(retrieved_docs, relevant_docs):
    mrr = 0
    for cur_retrieved_docs, cur_relevant_doc in zip(retrieved_docs, relevant_docs):
        for i, doc_id in enumerate(cur_retrieved_docs):
            if doc_id == cur_relevant_doc:
                mrr += 1 / (i + 1)
                break

    return mrr / len(relevant_docs)


# Initialize document store and retrievers
document_store = WeaviateDocumentStore(url="http://localhost:8080")
bm25_retriever = WeaviateBM25Retriever(document_store=document_store, top_k=100)
embedding_retriever = WeaviateEmbeddingRetriever(document_store=document_store, top_k=100)

# Load the dataset
comments_df = pd.read_csv("data/reddit_comments_sample_with_embeddings.csv")
comments_df["embedding"] = comments_df["embedding"].apply(literal_eval)
comments_df["arxiv_id"] = comments_df["arxiv_id"].astype(str)

# Retrieve articles
bm25_results = []
for post in tqdm(comments_df["clean_reddit_post"].values):
    retrieved_docs = bm25_retriever.run(query=post)["documents"]
    retrieved_ids = [doc.id for doc in retrieved_docs]
    bm25_results.append(retrieved_ids)

embedding_results = []
for emb in tqdm(comments_df["embedding"].values):
    retrieved_docs = embedding_retriever.run(query_embedding=emb)["documents"]
    retrieved_ids = [doc.id for doc in retrieved_docs]
    embedding_results.append(retrieved_ids)


# # BM25 results
bm25_recall_at_1 = recall(bm25_results, comments_df["arxiv_id"].values, top_k=1)
print(f"BM25 Recall@1: {bm25_recall_at_1}")
bm25_recall_at_10 = recall(bm25_results, comments_df["arxiv_id"].values, top_k=10)
print(f"BM25 Recall@10: {bm25_recall_at_10}")
bm25_recall_at_100 = recall(bm25_results, comments_df["arxiv_id"].values, top_k=100)
print(f"BM25 Recall@100: {bm25_recall_at_100}")
bm25_mrr = mean_reciprocal_rank(bm25_results, comments_df["arxiv_id"].values)
print(f"BM25 MRR: {bm25_mrr}")

# Embedding results
embedding_recall_at_1 = recall(embedding_results, comments_df["arxiv_id"].values, top_k=1)
print(f"Embedding Recall@1: {embedding_recall_at_1}")
embedding_recall_at_10 = recall(embedding_results, comments_df["arxiv_id"].values, top_k=10)
print(f"Embedding Recall@10: {embedding_recall_at_10}")
embedding_recall_at_100 = recall(embedding_results, comments_df["arxiv_id"].values, top_k=100)
print(f"Embedding Recall@100: {embedding_recall_at_100}")
embedding_mrr = mean_reciprocal_rank(embedding_results, comments_df["arxiv_id"].values)
print(f"Embedding MRR: {embedding_mrr}")
