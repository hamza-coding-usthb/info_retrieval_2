import os

def load_descriptor_or_index(query, search_type, token_method, stem_method, use_index):
    """
    Load data from descriptor or inverse index files and search for relevant terms or docs.

    Parameters:
    - query: List of terms or a single term/document ID to search for.
    - search_type: "terms_per_doc" or "docs_per_term".
    - token_method: "Split" or "Token".
    - stem_method: "" (no stemming), "Porter", or "Lancaster".
    - use_index: Boolean indicating whether to use the inverse index.

    Returns:
    - List of tuples representing the results.
    """
    print("use index: ", use_index)

    # Determine file name
    file_prefix = "Inverse" if use_index else "Descriptor"
    filename = f"{file_prefix}{token_method}"
    if stem_method:
        filename += stem_method
    filename += ".txt"
    
    # Ensure file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    # Ensure query is a list for iteration
    if not isinstance(query, list):
        query = [query]

    # Load file into a structured format
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            
            if use_index:
                term, doc_id, freq, weight = parts
                for q in query:
                    
                    if search_type == "docs_per_term" and term == q:
                        data.append((term, doc_id, int(freq), float(weight)))
                    elif search_type == "terms_per_doc" and doc_id == q:
                        data.append((doc_id, term, int(freq), float(weight)))
            else:
                doc_id, term, freq, weight = parts
                for q in query:
                    
                    if search_type == "docs_per_term" and term == q:
                        data.append((term, doc_id, int(freq), float(weight)))
                    elif search_type == "terms_per_doc" and doc_id == q:
                        data.append((doc_id, term, int(freq), float(weight)))
    
    
    return data
