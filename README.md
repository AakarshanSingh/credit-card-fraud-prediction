# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using various classification algorithms and advanced techniques like hyperparameter tuning.

## Overview

This project uses a credit card fraud dataset to build and compare multiple machine learning models for fraud detection. The dataset contains transactions made by credit cards in September 2013 by European cardholders, where fraudulent transactions represent only 0.172% of all transactions, making it a highly imbalanced dataset.

## Features

- **Data Preprocessing**: Outlier handling using IQR method, feature scaling with MinMaxScaler
- **Class Imbalance Handling**: Under-sampling of legitimate transactions to create a balanced dataset
- **Multiple ML Models**: Comparison of 11 different algorithms including:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - Multi-Layer Perceptron (MLP)
  - Stochastic Gradient Descent (SGD)
  - XGBoost
  - LightGBM
  - CatBoost

- **Hyperparameter Tuning**: GridSearchCV optimization for best performing models
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and confusion matrix
- **Visualization**: Correlation heatmap for feature analysis

## Dataset

The dataset is not included in this repository due to its size. Please download it from Google Drive:

**Dataset Download**: [Credit Card Fraud Dataset](https://drive.google.com/drive/folders/19HYf7L2-VdJi_T5iws39RJlGbGNCB5k_?usp=drive_link)

After downloading, place the `creditcard.csv` file in the `dataset/` folder.

### Dataset Structure
```
dataset/
    creditcard.csv
```

The dataset contains:
- **284,807 transactions** with 31 features
- **492 fraudulent transactions** (0.172%)
- **284,315 legitimate transactions** (99.828%)
- Features V1-V28 are PCA-transformed for anonymity
- Additional features: Time, Amount, and Class (target variable)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AakarshanSingh/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset from the Google Drive link above and place it in the `dataset/` folder.

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook credit-card-fraud-prediction.ipynb
```

2. Run all cells sequentially to:
   - Load and explore the data
   - Preprocess the dataset
   - Train multiple models
   - Compare model performances
   - Perform hyperparameter tuning
   - Evaluate on the full dataset

## Model Performance

The project compares multiple algorithms and performs hyperparameter tuning on the best performers (CatBoost and XGBoost). The final evaluation includes:

- **Weighted metrics** for overall performance
- **Fraud-specific metrics** for minority class detection
- **Confusion matrix** for detailed error analysis
- **Classification report** for per-class performance

## Key Results

- **Best Model**: CatBoost with optimized hyperparameters
- **Evaluation**: Comprehensive testing on both balanced training data and full imbalanced dataset
- **Metrics**: High precision and recall for fraud detection while maintaining overall accuracy

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and evaluation
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Light gradient boosting machine
- **CatBoost**: Categorical boosting algorithm
- **Seaborn/Matplotlib**: Data visualization
- **Jupyter Notebook**: Interactive development environment


## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset source: Machine Learning Group - ULB (Université Libre de Bruxelles)
- Original dataset available on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)