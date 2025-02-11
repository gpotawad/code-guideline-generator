import ast
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Sample Python code for analysis
code = """
# This is a comment
def hello():
    print("Hello, World!")  # Inline comment
"""

# Step 1: Extract Comments Using Regex
comments = re.findall(r"#.*", code)
print("Extracted Comments:", comments)

# Step 2: Parse Code Using AST
tree = ast.parse(code)
functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
print("Functions Found:", functions)

# Step 3: NLP with NLTK (Tokenization)
nltk.download('punkt')
tokens = nltk.word_tokenize("This is a sample NLP test.")
print("Tokenized Text:", tokens)

# Step 4: Basic Feature Extraction with scikit-learn
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["This is a test.", "Another test here."])
print("Feature Names:", vectorizer.get_feature_names_out())
