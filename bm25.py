import math
import pandas as pd
from collections import defaultdict

def compute_bm25(query_terms, descriptor_file, inverse_file, useIndex, k=1.5, b=0.75):
    """
    Compute BM25 scores for documents given a query.

    Parameters:
    - query_terms: List of terms in the query.
    - descriptor_file: Path to the descriptor file.
    - inverse_file: Path to the inverse index file.
    - useIndex: Boolean indicating whether to use the inverse index.
    - k: BM25 parameter for term frequency scaling.
    - b: BM25 parameter for document length normalization.

    Returns:
    - A dictionary with document IDs as keys and BM25 scores as values.
    """
    print(query_terms)
    if useIndex:
        print("file selected for bm25:", inverse_file)
        # Load data from the inverse index file
        df = pd.read_csv(descriptor_file, sep=r"\s+", names=["term", "doc_id", "freq", "weight"], engine="python")

        # Precompute document lengths and average document length
        doc_lengths = df.groupby("doc_id")["freq"].sum()
        avg_doc_len = doc_lengths.mean()
        N = df["doc_id"].nunique()

        # Debug: Print document statistics
        print(f"Total documents (N): {N}")
        print(f"Average document length (avg_doc_len): {avg_doc_len}")
        print(f"Document lengths (doc_lengths):\n{doc_lengths}")

        bm25_scores = defaultdict(float)

        # Process each query term
        for term in query_terms:
            term = term.lower().strip()
            print(f"\nProcessing term: {term}")

            # Filter data for the current term
            term_data = df[df["term"].str.lower() == term]

            if term_data.empty:
                print(f"Term '{term}' not found in any document.")
                continue

            # Number of documents containing the term
            n_i = term_data["doc_id"].nunique()
            print(f"Number of documents containing '{term}' (n_i): {n_i}")

            # Corrected IDF Calculation to Match Descriptor Logic
            idf = math.log(((N - n_i + 0.5) / (n_i + 0.5)), 10)  # No "+1" added to IDF
            print(f"IDF for '{term}': {idf}")

            # Process each document containing the term
            for _, row in term_data.iterrows():
                doc_id = row["doc_id"]
                freq_td = row["freq"]
                doc_len = doc_lengths[doc_id]

                # Corrected BM25 Term Frequency Weighting to Match Descriptor Logic
                numerator = freq_td
                denominator = freq_td + k * ((1 - b) + b * (doc_len / avg_doc_len))
                score = idf * (numerator / denominator)

                # Debug: Print per-document computations
                print(f"Document ID: {doc_id}")
                print(f"Term frequency in doc (freq_td): {freq_td}")
                print(f"Document length (doc_len): {doc_len}")
                print(f"Numerator: {numerator}")
                print(f"Denominator: {denominator}")
                print(f"Score for term '{term}' in doc '{doc_id}': {score}")

                bm25_scores[doc_id] += score

    else:
        print("file selected for bm25:", descriptor_file)
        # Descriptor file logic remains untouched
        df = pd.read_csv(descriptor_file, sep=r"\s+", names=["doc_id", "term", "freq", "weight"], engine="python")

        # Precompute document lengths and average document length
        doc_lengths = df.groupby("doc_id")["freq"].sum()
        avg_doc_len = doc_lengths.mean()
        N = df["doc_id"].nunique()

        # Debug: Print document statistics
        print(f"Total documents (N): {N}")
        print(f"Average document length (avg_doc_len): {avg_doc_len}")
        print(f"Document lengths (doc_lengths):\n{doc_lengths}")

        bm25_scores = defaultdict(float)

        # Process each query term
        for term in query_terms:
            term = term.lower().strip()
            print(f"\nProcessing term: {term}")

            term_data = df[df["term"] == term]

            if term_data.empty:
                print(f"Term '{term}' not found in any document.")
                continue

            # Number of documents containing the term
            n_i = term_data["doc_id"].nunique()
            print(f"Number of documents containing '{term}' (n_i): {n_i}")

            # Compute IDF
            idf = math.log(((N - n_i + 0.5) / (n_i + 0.5)), 10)
            print(f"IDF for '{term}': {idf}")

            # Process each document containing the term
            for _, row in term_data.iterrows():
                doc_id = row["doc_id"]
                freq_td = row["freq"]
                doc_len = doc_lengths[doc_id]

                # Compute BM25 term weight
                numerator = freq_td 
                denominator = freq_td + k * ((1 - b) + b * (doc_len / avg_doc_len))
                score = idf * (numerator / denominator)

                # Debug: Print per-document computations
                print(f"Document ID: {doc_id}")
                print(f"Term frequency in doc (freq_td): {freq_td}")
                print(f"Document length (doc_len): {doc_len}")
                print(f"Numerator: {numerator}")
                print(f"Denominator: {denominator}")
                print(f"Score for term '{term}' in doc '{doc_id}': {score}")

                bm25_scores[doc_id] += score

    # Return scores sorted by relevance
    sorted_scores = dict(sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True))

    # Debug: Print final scores
    print("\nFinal BM25 Scores:")
    for doc_id, score in sorted_scores.items():
        print(f"Document ID: {doc_id}, Score: {score}")

    return sorted_scores
