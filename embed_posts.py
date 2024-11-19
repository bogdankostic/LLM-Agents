from sentence_transformers import SentenceTransformer
import pandas as pd


BATCH_SIZE = 16

# Initialize the SentenceTransformer model
embedder = SentenceTransformer(
    model_name_or_path="BAAI/bge-small-en-v1.5",
    device="mps"
)
# Load the dataset
posts_df = pd.read_csv("data/reddit_comments_sample.csv")

# Embed the posts
embeddings = embedder.encode(
    posts_df["clean_reddit_post"].tolist(),
    normalize_embeddings=True,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

# Save the embeddings
posts_df["embedding"] = embeddings.tolist()
posts_df.to_csv("data/reddit_comments_sample_with_embeddings.csv", index=False)
