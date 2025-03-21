# Machine Learning Coding Project

## Project Overview
This project implements five core machine learning algorithms (Linear Regression, Logistic Regression, Decision Trees, K-Nearest Neighbors, and Naive Bayes) from scratch to solve classification tasks across five real-world datasets. The models are evaluated on medical diagnosis, food safety, industrial monitoring, vehicle safety, and game analytics problems.

### Key Features
- Algorithms implemented without ML libraries
- Dataset-specific preprocessing
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Comparative visualizations (confusion matrices, accuracy bars)

---

## How to Run the Code

### Prerequisites
- Google account (for Colab) **or** Jupyter Notebook
- Python 3.8+ (pre-installed in Colab)

### Step-by-Step Instructions

1. **Open the Notebook**
   - Preferred Method:  
     [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14vktqrHtOfIXVtwHvjB5ixt61c690BCd?usp=sharing)
   - Alternative Method:  
     Upload `Coding_Project.ipynb` to [Jupyter Notebook](https://jupyter.org/)

2. **Upload Algorithm Files**
   ```python
   # In Colab:
   from google.colab import files
   uploaded = files.upload()
   
   # Upload these files:
   # - LinearRegression.py
   # - LogisticRegression.py
   # - DecisionTree.py
   # - KNN.py
   # - BayesianClassifier.py

Run the Notebook

Execute cells sequentially using:

Shift + Enter (Colab/Jupyter)

Runtime > Run All (Colab menu)

Datasets are automatically loaded from URLs:

python
Copy
# Example dataset URL (Breast Cancer):
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
View Results

Automatic visualizations after each dataset run

Final results compiled in Results.pdf

Dependencies:
numpy, pandas, matplotlib, seaborn, scikit-learn (metrics only)
Dataset Sources: All loaded from UCI ML Repository URLs

Support
For issues/questions:
✉️ Contact: [Your Email]
📌 GitHub Issues: [Project Repository Issues Page]

Note: Results may vary slightly (±2%) due to random train-test splits.
