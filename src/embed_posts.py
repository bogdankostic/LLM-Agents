import argparse

from sentence_transformers import SentenceTransformer
import pandas as pd


BATCH_SIZE = 16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--column_names", type=str, nargs="+")
    args = parser.parse_args()

    file_path = args.file_path
    column_names = args.column_names

    # Initialize the SentenceTransformer model
    embedder = SentenceTransformer(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        device="mps"
    )
    # Load the dataset
    posts_df = pd.read_csv(file_path)

    for column_name in column_names:
        # Embed the posts
        embeddings = embedder.encode(
            posts_df[column_name].tolist(),
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )

        # Add the embeddings to the DataFrame
        posts_df[f"{column_name}_embedding"] = embeddings.tolist()

    # Save the embeddings
    filename = file_path.split(".")[0]
    posts_df.to_csv(f"{filename}_embeddings.csv", index=False)

