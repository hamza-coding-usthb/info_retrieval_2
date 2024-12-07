import math
import pandas as pd
import numpy as np

def load_data(filename, use_index=False):
    """
    Load descriptor or inverse index data using Pandas.

    Parameters:
    - filename: Path to the file.
    - use_index: Boolean indicating whether to load the inverse index.

    Returns:
    - A Pandas DataFrame containing the structured data.
    """
    if use_index:
        return pd.read_csv(filename, sep=r"\s+", names=["term", "doc_id", "freq", "weight"])
    else:
        return pd.read_csv(filename, sep=r"\s+", names=["doc_id", "term", "freq", "weight"])


def scalar_product(query_terms, query_weights, inverse_df):
    scores = pd.Series(dtype=float)

    for term in query_terms:
        if term in query_weights and term in inverse_df["term"].values:
            query_weight = query_weights[term]
            term_data = inverse_df[inverse_df["term"] == term].copy()  # Ensure a copy
            term_data["score"] = query_weight * term_data["weight"]
            scores = scores.add(term_data.groupby("doc_id")["score"].sum(), fill_value=0)

    return scores.sort_values(ascending=False)


def cosine_similarity(query_terms, query_weights, df):
    """
    Compute Cosine Similarity using a vectorized approach.

    Parameters:
    - query_terms: List of query terms.
    - query_weights: Dictionary of query term weights (binary: 1 if term in query, 0 otherwise).
    - inverse_df: Pandas DataFrame of the inverse index.

    Returns:
    - A Pandas Series of document scores sorted in descending order.
    """
    # Get unique document IDs and terms
    unique_docs = df["doc_id"].unique()
    unique_terms = df["term"].unique()

    # Create term-document matrix
    term_doc_matrix = pd.DataFrame(0, index=unique_terms, columns=unique_docs)

    for _, row in df.iterrows():
        term_doc_matrix.at[row["term"], row["doc_id"]] = row["weight"]

    print("\nTerm-Document Matrix:\n", term_doc_matrix)

    # Create query vector
    query_vector = pd.Series(0, index=unique_terms)
    for term in query_terms:
        if term in query_weights:
            query_vector[term] = query_weights[term]

    print("\nQuery Vector:\n", query_vector)

    # Compute dot products
    dot_products = term_doc_matrix.T.dot(query_vector)
    print("\nDot Products:\n", dot_products)

    # Compute magnitudes
    query_magnitude = np.sqrt((query_vector**2).sum())
    doc_magnitudes = np.sqrt((term_doc_matrix**2).sum(axis=0))
    print("\nQuery Magnitude:", query_magnitude)
    print("\nDocument Magnitudes:\n", doc_magnitudes)

    # Compute cosine similarity
    cosine_scores = dot_products / (query_magnitude * doc_magnitudes)
    cosine_scores = cosine_scores.dropna()  # Remove NaN results

    # Filter out zero scores
    cosine_scores = cosine_scores[cosine_scores > 0]

    print("\nFiltered Cosine Similarity Scores (Non-Zero):\n", cosine_scores)

    # Sort scores in descending order
    return cosine_scores.sort_values(ascending=False)

def cosine_similarity_2(query_terms, query_weights, df):
    """
    Compute RSV using Cosine Similarity with debugging.

    Parameters:
    - query_terms: List of query terms.
    - query_weights: Dictionary of query term weights (binary: 1 if term in query, 0 otherwise).
    - inverse_df: Pandas DataFrame of the inverse index.

    Returns:
    - A Pandas Series of document scores sorted in descending order.
    """
    scores = pd.Series(dtype=float)
    doc_magnitudes = pd.Series(dtype=float)

    # Precompute query magnitude
    query_magnitude = math.sqrt(sum(weight**2 for weight in query_weights.values()))
    print(f"Query Magnitude: {query_magnitude}")

    # Iterate through terms in the query
    for term in query_terms:
        print(f"\nProcessing term: {term}")
        if term in query_weights and term in df["term"].values:
            query_weight = query_weights[term]
            print(f"Query weight for term '{term}': {query_weight}")

            # Subset the inverse index for the term
            term_data = df[df["term"] == term].copy()
            print(f"Documents containing '{term}':\n{term_data}")

            # Calculate dot product components
            term_data["dot_product"] = query_weight * term_data["weight"]
            print(f"Dot product for '{term}':\n{term_data[['doc_id', 'dot_product']]}")

            scores = scores.add(term_data.groupby("doc_id")["dot_product"].sum(), fill_value=0)
            print(f"Scores after processing '{term}':\n{scores}")

            # Calculate squared weights for document magnitude
            term_data["doc_weight_sq"] = term_data["weight"] ** 2
            print(f"Document weight squared for '{term}':\n{term_data[['doc_id', 'doc_weight_sq']]}")

            doc_magnitudes = doc_magnitudes.add(term_data.groupby("doc_id")["doc_weight_sq"].sum(), fill_value=0)
            print(f"Document magnitudes after processing '{term}':\n{doc_magnitudes}")

    # Finalize document magnitudes
    doc_magnitudes = doc_magnitudes.apply(math.sqrt)
    print(f"\nFinal Document Magnitudes:\n{doc_magnitudes}")

    # Normalize scores using query and document magnitudes
    for doc_id in scores.index:
        if query_magnitude > 0 and doc_magnitudes.get(doc_id, 0) > 0:
            scores[doc_id] /= query_magnitude * doc_magnitudes[doc_id]
        else:
            scores[doc_id] = 0.0  # Avoid division by zero
        print(f"Normalized score for Document {doc_id}: {scores[doc_id]}")

    # Filter and sort results
    scores = scores[scores > 0].sort_values(ascending=False)
    print("\nFinal Cosine Similarity Scores:")
    print(scores)

    return scores



def weighted_jaccard_index(query_weights, df):
    """
    Compute RSV using Weighted Jaccard Index.

    Parameters:
    - query_weights: Dictionary of query term weights.
    - descriptor_df: Pandas DataFrame of the descriptor data.

    Returns:
    - A Pandas Series of document scores (excluding documents with zero RSV).
    """
    scores = pd.Series(dtype=float)

    # Group data by document ID
    for doc_id, doc_data in df.groupby("doc_id"):
        doc_data = doc_data.copy()  # Ensure a copy to avoid SettingWithCopyWarning

        # Create a Series of weights for the terms in the document
        doc_terms = doc_data.set_index("term")["weight"]
        dot_product = sum(query_weights.get(term, 0) * doc_terms.get(term, 0) for term in query_weights)

        # Compute magnitudes
        query_magnitude = sum(weight**2 for weight in query_weights.values())
        doc_magnitude = sum(weight**2 for weight in doc_terms.values)

        # Compute the weighted union
        weighted_union = query_magnitude + doc_magnitude - dot_product

        # Avoid division by zero and compute the score
        if weighted_union > 0:
            scores[doc_id] = dot_product / weighted_union

    # Filter out documents with zero RSV scores
    scores = scores[scores > 0]

    return scores.sort_values(ascending=False)


def compute_rsv(query, file, method, useIndex):
    """
    Compute RSV scores for a given query using the selected method.

    Parameters:
    - query: List of query terms.
    - descriptor_file: Path to the descriptor file.
    - inverse_file: Path to the inverse index file.
    - method: Selected RSV method ("Scalar Product", "Cosine Similarity", "Weighted Jaccard Index").

    Returns:
    - A Pandas Series of document scores.
    """
    if useIndex:
        df = load_data(file, use_index=True)
    else:
        df = load_data(file, use_index=False)
    query_weights = compute_query_weights(query)

    if method == "Scalar Product":
        return scalar_product(query, query_weights, df)
    elif method == "Cosine Similarity":
        return cosine_similarity(query, query_weights, df)
    elif method == "Weighted Jaccard Index":
        return weighted_jaccard_index(query_weights, df)
    else:
        raise ValueError(f"Unknown RSV method: {method}")


def compute_query_weights(query_terms):
    """
    Compute binary term weights for the query.

    Parameters:
    - query_terms: List of query terms.

    Returns:
    - A dictionary of term weights (1 if term is in the query, 0 otherwise).
    """
    return {term: 1 for term in query_terms}
