import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QRadioButton, QCheckBox, QLabel, QGroupBox, QComboBox, QMessageBox
)
import pandas as pd
from search import load_descriptor_or_index 
from RSV_methods import compute_rsv 
from bm25 import compute_bm25  # Import BM25 logic
from boolean_model import validate_boolean_query, evaluate_boolean_query_with_pandas

class SearchInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Search Interface with RSV Calculation and BM25")
        self.setGeometry(100, 100, 1000, 600)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Query Section (Horizontal)
        query_layout = QHBoxLayout()
        query_label = QLabel("Query:")
        self.query_input = QLineEdit()
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_search)

        query_layout.addWidget(query_label)
        query_layout.addWidget(self.query_input)
        query_layout.addWidget(search_button)
        main_layout.addLayout(query_layout)

        # Tokenization, Stemming, and Search Type (Single Horizontal Line)
        processing_layout = QHBoxLayout()

        # Tokenization Group
        tokenization_group = QGroupBox("Tokenization")
        tokenization_layout = QVBoxLayout()
        self.split_rb = QRadioButton("Split")
        self.split_rb.setChecked(True)
        self.token_rb = QRadioButton("Token")
        tokenization_layout.addWidget(self.split_rb)
        tokenization_layout.addWidget(self.token_rb)
        tokenization_group.setLayout(tokenization_layout)

        processing_layout.addWidget(tokenization_group)
        # Add a "Use Index" checkbox to the processing layout
        self.use_index_checkbox = QCheckBox("Use Inverse Files")
        processing_layout.addWidget(self.use_index_checkbox)
        # Stemming Group
        stemming_group = QGroupBox("Stemming")
        stemming_layout = QVBoxLayout()
        self.no_stem_rb = QRadioButton("No Stem")
        self.no_stem_rb.setChecked(True)
        self.porter_stem_rb = QRadioButton("Porter Stemmer")
        self.lancaster_stem_rb = QRadioButton("Lancaster Stemmer")
        stemming_layout.addWidget(self.no_stem_rb)
        stemming_layout.addWidget(self.porter_stem_rb)
        stemming_layout.addWidget(self.lancaster_stem_rb)
        stemming_group.setLayout(stemming_layout)

        processing_layout.addWidget(stemming_group)

        # Search Type Group
        search_type_group = QGroupBox("Search Type")
        search_type_layout = QVBoxLayout()
        self.terms_per_doc_rb = QRadioButton("Terms per Doc")
        self.terms_per_doc_rb.setChecked(True)
        self.docs_per_term_rb = QRadioButton("Docs per Term")
        search_type_layout.addWidget(self.terms_per_doc_rb)
        search_type_layout.addWidget(self.docs_per_term_rb)
        search_type_group.setLayout(search_type_layout)

        processing_layout.addWidget(search_type_group)

        main_layout.addLayout(processing_layout)

        # RSV and BM25 Blocks (Single Horizontal Line)
        combined_rsv_bm25_layout = QHBoxLayout()

        # RSV Section
        rsv_group = QGroupBox("RSV Options")
        rsv_layout = QVBoxLayout()
        self.rsv_checkbox = QCheckBox("Activate RSV Calculation")
        self.rsv_method_dropdown = QComboBox()
        self.rsv_method_dropdown.addItems(["Scalar Product", "Cosine Similarity", "Weighted Jaccard Index"])
        rsv_layout.addWidget(self.rsv_checkbox)
        rsv_layout.addWidget(self.rsv_method_dropdown)
        rsv_group.setLayout(rsv_layout)
        # Boolean Model Section
        boolean_group = QGroupBox("Boolean Model")
        boolean_layout = QVBoxLayout()
        self.boolean_checkbox = QCheckBox("Use Boolean Model")
        self.boolean_checkbox.stateChanged.connect(self.toggle_boolean_model)

        boolean_layout.addWidget(self.boolean_checkbox)
        boolean_group.setLayout(boolean_layout)
        combined_rsv_bm25_layout.addWidget(rsv_group)
        # Add Boolean Model section inline with RSV and BM25
        combined_rsv_bm25_layout.addWidget(boolean_group)
        # BM25 Section
        bm25_group = QGroupBox("BM25 Options")
        bm25_layout = QVBoxLayout()
        self.bm25_checkbox = QCheckBox("Use BM25")
        self.bm25_checkbox.stateChanged.connect(self.toggle_rsv_methods)

        bm25_params_layout = QHBoxLayout()
        k_label = QLabel("k:")
        self.k_input = QLineEdit("1.5")  # Default value
        b_label = QLabel("b:")
        self.b_input = QLineEdit("0.75")  # Default value

        bm25_params_layout.addWidget(k_label)
        bm25_params_layout.addWidget(self.k_input)
        bm25_params_layout.addWidget(b_label)
        bm25_params_layout.addWidget(self.b_input)

        bm25_layout.addWidget(self.bm25_checkbox)
        bm25_layout.addLayout(bm25_params_layout)
        bm25_group.setLayout(bm25_layout)

        combined_rsv_bm25_layout.addWidget(bm25_group)

        main_layout.addLayout(combined_rsv_bm25_layout)

        # Results Table
        self.results_table = QTableWidget()
        main_layout.addWidget(self.results_table)
    def toggle_boolean_model(self):
        """Enable or disable RSV and BM25 methods based on the Boolean model checkbox."""
        enabled = not self.boolean_checkbox.isChecked()
        self.rsv_checkbox.setEnabled(enabled)
        self.rsv_method_dropdown.setEnabled(enabled)
        self.bm25_checkbox.setEnabled(enabled)
        self.k_input.setEnabled(enabled)
        self.b_input.setEnabled(enabled)
    def toggle_rsv_methods(self):
        """Enable or disable RSV methods based on the BM25 checkbox."""
        enabled = not self.bm25_checkbox.isChecked()
        self.rsv_checkbox.setEnabled(enabled)
        self.rsv_method_dropdown.setEnabled(enabled)

    def perform_search(self):
        query = self.query_input.text().strip()
        query_bool = query
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a query.")
            return

        try:
            # Determine the file based on the interface selections
            use_index = self.use_index_checkbox.isChecked()  # Check if "Use Inverse Files" is ticked
            file_prefix = "Inverse" if use_index else "Descriptor"
            tokenization = "Token" if self.token_rb.isChecked() else "Split"

            if self.no_stem_rb.isChecked():
                stemming = ""
            elif self.porter_stem_rb.isChecked():
                stemming = "Porter"
            elif self.lancaster_stem_rb.isChecked():
                stemming = "Lancaster"
            else:
                stemming = ""

            # Construct the file name
            selected_file = f"{file_prefix}{tokenization}{stemming}.txt"
            
            # Load the dataset
            df = pd.read_csv(selected_file, delim_whitespace=True, header=None, names=["doc_id", "term", "freq", "weight"])
            term_to_docs = df.groupby("term")["doc_id"].apply(set).to_dict()

            # Preprocess the query if necessary
            if stemming or tokenization == "Token":
                import re
                from nltk.corpus import stopwords

                stop_words = set(stopwords.words("english"))
                query_tokens = re.findall(r"\w+", query.lower())  # Basic tokenization with regex
                query_tokens = [word for word in query_tokens if word not in stop_words]  # Remove stopwords

                if stemming:
                    if stemming == "Porter":
                        from nltk.stem import PorterStemmer
                        stemmer = PorterStemmer()
                    elif stemming == "Lancaster":
                        from nltk.stem import LancasterStemmer
                        stemmer = LancasterStemmer()
                    query_tokens = [stemmer.stem(word) for word in query_tokens]

                query = query_tokens  # Preprocessed query as a list of terms
            else:
                query = query.split()  # Default split

            # Perform the selected search method
            if self.boolean_checkbox.isChecked():
                # Boolean Model (no query preprocessing)
                if not validate_boolean_query(query_bool):
                    QMessageBox.critical(self, "Error", "Invalid Boolean query.")
                    return

                results = evaluate_boolean_query_with_pandas(query_bool, term_to_docs)
                self.display_boolean_results(results)

            elif self.bm25_checkbox.isChecked():
                # BM25 search
                print("did it reach here?")
                k = float(self.k_input.text())
                b = float(self.b_input.text())
                bm25_scores = compute_bm25(query, selected_file, None,use_index ,k, b )
                self.display_rsv_results(bm25_scores, "BM25")

            elif self.rsv_checkbox.isChecked():
                # RSV search
                rsv_method = self.rsv_method_dropdown.currentText()
                rsv_scores = compute_rsv(query, selected_file, rsv_method, use_index)
                self.display_rsv_results(rsv_scores, rsv_method)

            else:
                # Descriptor/Inverse index search
                search_type = "terms_per_doc" if self.terms_per_doc_rb.isChecked() else "docs_per_term"

                results = load_descriptor_or_index(query, search_type, tokenization, stemming, use_index)
                self.display_index_results(results, search_type)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during calculation: {e}")







    def display_rsv_results(self, results, method):
        """
        Display the results from RSV or BM25 in the results table.

        Parameters:
        - results: Pandas Series with document IDs as the index and scores as values.
        - method: The method name (e.g., "BM25", "Scalar Product").
        """
        # Configure table columns
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Document", f"{method} Score"])

        # Populate table rows
        self.results_table.setRowCount(len(results))
        for row_idx, (doc_id, score) in enumerate(results.items()):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(doc_id)))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{score:.4f}"))  # Format score to 4 decimal places
    def display_boolean_results(self, results):
        """
        Display the results from the Boolean model in the results table.

        Parameters:
        - results: Set of document IDs matching the query.
        """
        self.results_table.setColumnCount(1)
        self.results_table.setHorizontalHeaderLabels(["Document"])

        self.results_table.setRowCount(len(results))
        for row_idx, doc_id in enumerate(sorted(results)):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(doc_id)))


    def display_index_results(self, results, search_type):
        """
        Display the results from the descriptor or inverse file search.

        Parameters:
        - results: List of tuples representing the search results.
        - search_type: The type of search performed ("terms_per_doc" or "docs_per_term").
        """
        if not results:
            QMessageBox.information(self, "Info", "No results found.")
            return

        # Configure table columns based on the search type
        if search_type == "terms_per_doc":
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels(["Document", "Term", "Frequency", "Weight"])
        elif search_type == "docs_per_term":
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels(["Term", "Document", "Frequency", "Weight"])
        else:
            QMessageBox.critical(self, "Error", f"Unknown search type: {search_type}")
            return

        # Populate table rows
        self.results_table.setRowCount(len(results))
        for row_idx, result in enumerate(results):
            for col_idx, value in enumerate(result):
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchInterface()
    window.show()
    sys.exit(app.exec_())
