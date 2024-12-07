import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
import math
from nltk.probability import FreqDist  # Import FreqDist

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

# Define paths for documents
docs = [f'Collections/D{i}.txt' for i in range(1, 7)]
N = len(docs)  # Total number of documents

# Define the two tokenization methods
tokenizer_split = lambda text: text.split()
tokenizer_regex = RegexpTokenizer(r'\b\d+(?:\.\d+)?(?:x\d+)?\b|\b\w+(?:-\w+)*\b').tokenize

# Define the two stemming methods
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

# Load English stop words
stop_words = set(stopwords.words('english'))

# Utility to write data to a file
def write_to_file(filename, data, mode='w'):
    with open(filename, mode) as f:
        for line in sorted(data):  # Sort data before writing
            f.write(' '.join(map(str, line)) + '\n')

# Function to calculate frequencies
def calculate_frequencies(text, tokenizer, stop_words, stemmer=None):
    tokens = tokenizer(text)
    tokens = [t for t in tokens if t not in stop_words]  # Remove stop words
    if stemmer:
        tokens = [stemmer(t) for t in tokens]
    return FreqDist(tokens)

# Initialize dictionaries for inverse indexing and document frequencies
inverse_indices = {
    'Split': {}, 'Token': {}, 'SplitPorter': {}, 'TokenPorter': {}, 'SplitLancaster': {}, 'TokenLancaster': {}
}
doc_frequencies = {
    'Split': {}, 'Token': {}, 'SplitPorter': {}, 'TokenPorter': {}, 'SplitLancaster': {}, 'TokenLancaster': {}
}

# ----------------------------
# Step 1: Process documents to generate token files and document frequencies
# ----------------------------
for doc_name in docs:
    doc_number = int(doc_name.split('/')[-1][1])  # Extract the document number as an integer

    with open(doc_name, 'r') as f:
        text = f.read().lower()  # Read and lowercase text

    # Apply tokenization and stemming combinations
    token_methods = {'Split': tokenizer_split, 'Token': tokenizer_regex}
    stem_methods = {'': None, 'Porter': porter_stemmer.stem, 'Lancaster': lancaster_stemmer.stem}

    for token_name, tokenizer in token_methods.items():
        for stem_name, stem_func in stem_methods.items():
            # Calculate frequencies
            frequencies = calculate_frequencies(text, tokenizer, stop_words, stem_func)
            max_freq = max(frequencies.values()) if frequencies else 1

            # Update document frequencies for the current term
            key = f"{token_name}{stem_name}"
            for term in frequencies.keys():
                if term not in doc_frequencies[key]:
                    doc_frequencies[key][term] = set()
                doc_frequencies[key][term].add(doc_number)

            # Write tokens to descriptor files (weights will be calculated in the next step)
            descriptor_filename = f"Descriptor{key}.txt"
            descriptor_data = [(doc_number, term, freq, 0.0) for term, freq in frequencies.items()]
            write_to_file(descriptor_filename, descriptor_data, mode='a')

            # Initialize inverse index with frequencies and placeholders for weights
            for term, freq in frequencies.items():
                if term not in inverse_indices[key]:
                    inverse_indices[key][term] = []
                inverse_indices[key][term].append((doc_number, freq, 0.0))  # Placeholder for weight

# ----------------------------
# Step 2: Calculate weights and update descriptor files
# ----------------------------
# Clear descriptor files to rewrite with updated weights
for key in doc_frequencies.keys():
    open(f"Descriptor{key}.txt", 'w').close()  # Clear descriptor files

for doc_name in docs:
    doc_number = int(doc_name.split('/')[-1][1])  # Extract the document number as an integer

    with open(doc_name, 'r') as f:
        text = f.read().lower()  # Read and lowercase text

    # Apply tokenization and stemming combinations
    token_methods = {'Split': tokenizer_split, 'Token': tokenizer_regex}
    stem_methods = {'': None, 'Porter': porter_stemmer.stem, 'Lancaster': lancaster_stemmer.stem}

    for token_name, tokenizer in token_methods.items():
        for stem_name, stem_func in stem_methods.items():
            # Recalculate frequencies to get max frequency
            frequencies = calculate_frequencies(text, tokenizer, stop_words, stem_func)
            max_freq = max(frequencies.values()) if frequencies else 1

            # Calculate weights
            key = f"{token_name}{stem_name}"
            updated_weights = []
            for term, freq in frequencies.items():
                ni = len(doc_frequencies[key].get(term, set()))  # Number of documents containing the term
                weight = (freq / max_freq) * math.log10(((N / ni) + 1))  # Calculate weight

                # Add updated weights to the list to write to descriptor file
                updated_weights.append((doc_number, term, freq, round(weight, 4)))

                # Update inverse index with calculated weights
                for entry in inverse_indices[key][term]:
                    if entry[0] == doc_number:
                        inverse_indices[key][term].remove(entry)
                        inverse_indices[key][term].append((doc_number, freq, round(weight, 4)))

            # Overwrite updated weights to descriptor files
            descriptor_filename = f"Descriptor{key}.txt"
            write_to_file(descriptor_filename, updated_weights, mode='a')  # Append mode for descriptors

# ----------------------------
# Write inverse indices to files after all documents have been processed
# ----------------------------
for key, index_data in inverse_indices.items():
    inverse_filename = f"Inverse{key}.txt"
    inverse_data = [(term, doc_number, freq, weight) for term, entries in index_data.items() for (doc_number, freq, weight) in entries]
    write_to_file(inverse_filename, inverse_data, mode='w')  # Overwrite mode for inverse files

print("Descriptors and inverse files with term weights created successfully.")
