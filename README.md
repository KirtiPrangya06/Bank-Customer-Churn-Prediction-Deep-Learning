# Bank Customer Churn Prediction Using Deep Learning

An end-to-end Jupyter notebook exploring bank customer churn through data visualization, feature engineering, and neural-network classification. The goal is to predict whether a customer will leave a bank and help prioritize retention efforts.

## Project overview

- **Task:** Binary classification of customer churn (`Exited`: 1 = churn, 0 = retained).
- **Data:** `bank-1.csv` — 10,000 customers with 14 original columns, including customer identifiers, demographics, account activity, and the churn label.
- **Workflow:** Exploratory data analysis → encoding and scaling → train/test split → neural-network experiments → evaluation → recommendations.
- **Evaluation:** Recall, accuracy, precision, F1-score, ROC-AUC, ROC curves, and confusion matrices.

## Approaches compared

1. Neural network with stochastic gradient descent (SGD).
2. Neural network with Adam optimizer.
3. Neural network with dropout and Adam.
4. Neural network with grid-search hyperparameter tuning.
5. Neural network with SMOTE oversampling and Adam.

## Results documented in the notebook

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NN with SGD | 0.808 | 0.732 | 0.519 | 0.608 | 0.846 |
| NN with Adam | 0.807 | 0.708 | 0.519 | 0.599 | 0.849 |
| NN with Dropout + Adam | 0.764 | **0.813** | 0.455 | 0.584 | **0.860** |
| NN with Grid Search Tuning + Adam | 0.793 | 0.754 | 0.494 | 0.597 | 0.854 |
| NN with SMOTE + Adam | **0.823** | 0.705 | **0.550** | **0.618** | 0.856 |

The notebook selects **dropout + Adam** for its higher reported recall (81.3%) and ROC-AUC (0.860). These values are *recorded from the supplied notebook* and were not independently reproduced as part of GitHub packaging.

## Getting started

Use **Python 3.10 or 3.11** in a fresh virtual environment.

```bash
# Clone after uploading to GitHub
git clone https://github.com/YOUR_USERNAME/Bank-Customer-Churn-Prediction-Deep-Learning.git
cd Bank-Customer-Churn-Prediction-Deep-Learning

# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# macOS / Linux: python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Bank_Customer_Churn_Prediction.ipynb
```

Run notebook cells from top to bottom. The notebook reads `bank-1.csv` from the working directory; launch Jupyter inside the repository folder. Hyperparameter grid search can take substantial CPU time.

## Repository contents

```text
Bank-Customer-Churn-Prediction-Deep-Learning/
├── Bank_Customer_Churn_Prediction.ipynb
├── bank-1.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Key observations

The notebook explores relationships between churn and factors such as customer age, geography, account activity, and number of bank products. These findings are descriptive associations, not proven causes.

## Evaluation caveat

The notebook selects ROC thresholds using the same test labels used to report model performance, which can lead to optimistic test metrics. For rigorous production evaluation, choose thresholds and tune models on training/validation folds, then reserve the held-out test set for one final assessment. The existing notebook and its reported results are preserved aside from removing runtime package-install commands.

## Dataset note

The dataset is included for local reproducibility. Before distributing it beyond this repository, confirm you have permission to publish the data and that it contains no confidential customer information. No specific external dataset license or provenance was supplied.

## License

No license has been assigned. Add a `LICENSE` file only if you have the right to license all included material.
