# Sales Prediction using Python

## Overview

Sales prediction is an important task for businesses to estimate future product sales based on advertising expenditures. This project uses Machine Learning to predict sales using advertising budgets spent on TV, Radio, and Newspaper advertisements.

The model is trained using Linear Regression and provides accurate sales forecasts that can help businesses make data-driven marketing decisions.

---

## Project Objectives

* Load and analyze advertising data.
* Perform Exploratory Data Analysis (EDA).
* Visualize relationships between advertising channels and sales.
* Build a Linear Regression model.
* Predict future sales based on advertising budgets.
* Evaluate model performance using standard regression metrics.

---

## Dataset

The dataset contains advertising expenditures across different media channels and corresponding sales figures.

### Features

| Feature   | Description                           |
| --------- | ------------------------------------- |
| TV        | Advertising budget spent on TV        |
| Radio     | Advertising budget spent on Radio     |
| Newspaper | Advertising budget spent on Newspaper |
| Sales     | Product sales (Target Variable)       |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Project Workflow

1. Import required libraries.
2. Load the dataset.
3. Perform data cleaning.
4. Conduct Exploratory Data Analysis.
5. Visualize correlations between features.
6. Split data into training and testing sets.
7. Train a Linear Regression model.
8. Make predictions.
9. Evaluate model performance.
10. Predict sales for new advertising budgets.

---

## Model Used

### Linear Regression

Linear Regression is a supervised machine learning algorithm used for predicting continuous values. It establishes a relationship between advertising expenditures and sales.

---

## Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Sample Results

| Metric   | Value        |
| -------- | ------------ |
| MAE      | ~1.46        |
| RMSE     | ~1.78        |
| R² Score | ~0.90 - 0.95 |

A high R² score indicates that the model explains most of the variability in sales.

---

## Project Structure

```text
Sales-Prediction/
│
├── Advertising.csv
├── sales_prediction.py
├── sales_predictions.csv
├── requirements.txt
├── README.md
└── images/
    ├── correlation_heatmap.png
    └── actual_vs_predicted.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/dondakhushi/Sales_Prediction_using_python.git
```

Move into the project directory:

```bash
cd Sales-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python sales_prediction.py
```

---

## Future Enhancements

* Support multiple regression models.
* Hyperparameter tuning.
* Deployment using Flask or Streamlit.
* Interactive sales forecasting dashboard.

---

## Conclusion

This project demonstrates how Machine Learning can be used to predict sales based on advertising expenditures. The insights generated can help organizations optimize marketing budgets and improve business performance.

---

## Author

Khushi Donda
