from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack import Document
from datasets import load_dataset
from tqdm import tqdm

# Load the dataset
arxiv_dataset = load_dataset("somewheresystems/dataclysm-arxiv")
# Initialize document store
document_store = WeaviateDocumentStore(url="http://localhost:8080")

# Write documents to the document store
documents = []
for article in tqdm(arxiv_dataset["train"]):
    cur_doc = Document(
        content=f"{article['title']} {article['abstract']}",
        id=article["id"],
        embedding=article["abstract-embeddings"][0],
        meta={"title": article["title"],
              "authors": article["authors"],
              "categories": article["categories"],
              "date": article["update_date"],
              "arxiv_id": article["id"],
              }
    )
    documents.append(cur_doc)

    if len(documents) % 100 == 0:
        document_store.write_documents(documents)
        documents = []

document_store.write_documents(documents)
