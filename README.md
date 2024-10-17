# Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques on a customer churn dataset. The dataset contains various customer attributes, including demographic information and subscription details.

## Table of Contents

- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Model Training and Prediction](#model-training-and-prediction)
- [Streamlit Application](#streamlit-application)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Customer churn refers to the loss of customers, and understanding the factors that contribute to churn can help businesses improve customer retention. This project utilizes several machine learning models to predict churn based on customer data.

## Data Overview

The dataset used in this project includes the following columns:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Subscription Type**: Type of subscription the customer has
- **Contract Length**: Length of the customer’s contract
- **Total Spend**: Total amount spent by the customer
- **Tenure**: Duration of the customer's relationship with the company
- **Usage Frequency**: Frequency of usage by the customer
- **Payment Delay**: Delay in payments
- **Churn**: Target variable indicating if the customer has churned (1) or not (0)

## Installation

To run this project, you'll need the following libraries:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `plotly`
- `streamlit`

You can install these libraries using pip:

```
pip install pandas matplotlib seaborn scikit-learn xgboost lightgbm plotly streamlit
```

## Usage

1. Clone the repository or download the Jupyter Notebook.
2. Ensure that the dataset files are in the `data/` directory.
3. Run the Jupyter Notebook to execute the analysis and model training.
4. To run the Streamlit application, navigate to the directory containing the Streamlit script and execute:

```
streamlit run <your_script_name.py>
```

## Data Analysis

The data analysis section includes the following visualizations:

1. **Bar chart of Churn by Subscription Type**: Shows the relationship between subscription type and churn.
2. **Histogram of Age Distribution**: Displays the age distribution of customers.
3. **Bar chart of Churn by Gender**: Illustrates churn rates based on gender.
4. **Box plot of Usage Frequency by Gender**: Compares usage frequency between genders.
5. **Line chart of Contract Length vs. Payment Delay**: Analyzes the relationship between contract length and payment delays.

## Model Training and Prediction

The project includes the following steps for model training:

1. **Feature Engineering**: One-hot encoding of categorical variables, creation of new features, and dropping unnecessary columns.
2. **Train-Test Split**: Splitting the dataset into training and testing sets.
3. **Model Initialization**: Setting up various machine learning models including Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.
4. **Model Training and Evaluation**: Training each model and evaluating their performance using accuracy and classification report.

## Streamlit Application

A Streamlit application is included in this project to provide an interactive way to visualize the data and model predictions. The application includes:

- **Data Preview**: Displays the customer churn dataset.
- **Visualizations**:
  - Churn by Subscription Type
  - Age Distribution
  - Churn by Gender
  - Average Usage Frequency by Gender
  - Average Payment Delay by Contract Length
- **Model Training and Evaluation**:
  - Logistic Regression model trained on the dataset.
  - Displays the model's accuracy.
  - Confusion matrix and classification report are shown for detailed evaluation.

## Results

After evaluating all models, the best model was identified as:

- **Logistic Regression with an accuracy of 0.8~**

## Conclusion

This project demonstrates the application of machine learning for customer churn prediction. Future improvements could involve tuning hyperparameters, feature selection, and exploring advanced techniques like ensemble methods to improve model performance.

Feel free to modify any parts further to align with your specific project details!
