# Superstore Discount Predictor — Machine Learning Documentation

This repository contains an AI-driven predictive model designed to calculate optimal order discounts. By leveraging historical transaction records, the model serves as a decision support system to help maintain healthy profit margins and optimize pricing strategies.

---

## 📖 Table of Contents
1. [Business Value & Goal](#1-business-value--goal)
2. [Dataset Overview](#2-dataset-overview)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
4. [Model Performance & Architecture](#4-model-performance--architecture)

---

## 1. Business Value & Goal

In retail operations, offering discounts is a delicate balancing act. Over-discounting erodes profit margins, while under-discounting can result in lost sales volume or customer churn. 

This project uses an **AI-driven predictive model** to analyze historical logistics, geographic destinations, product categories, and transaction financials. By feeding these parameters into the trained model, it calculates the historically expected discount level, ensuring sales operations prevent margins from slipping into unprofitable territories.

---

## 2. Dataset Overview

The model is trained on the standard **Superstore Dataset** (`Sample - Superstore.csv`). The pipeline reads and processes the following features to generate predictions:

### Categorical Features
* **Ship Mode:** `Standard Class`, `Second Class`, `First Class`, `Same Day` (Reflects urgency and delivery overheads).
* **Segment:** `Consumer`, `Corporate`, `Home Office` (Identifies buyer demographic and scale).
* **Region:** `Central`, `East`, `South`, `West` (Geographic shipping region).
* **State & City:** Geographic metrics covering unique US territories (Impacts local taxation, distribution costs, and logistics).
* **Category:** `Furniture`, `Office Supplies`, `Technology` (Defines high-level product margins).
* **Sub-Category:** Granular product identification (e.g., `Bookcases`, `Appliances`, `Phones`, `Chairs`).

### Numerical Features
* **Sales ($):** Total transaction value before the discount is applied.
* **Quantity:** Total count of items ordered in a single transaction.
* **Profit ($):** Net profit earned from the sale, utilized to understand baseline historical margins.

---

## 3. Machine Learning Pipeline

The machine learning workflow is bundled into a structured, reproducible scikit-learn `Pipeline` object. This architecture ensures that raw data inputs are seamlessly transformed before being passed to the predictive estimator.



### Stage 1: Preprocessing (`ColumnTransformer`)
Features are automatically split by data type and processed through isolated pipelines:

* **Numerical Data Path:**
    1.  **`StandardScaler`:** Standardizes numeric features (`Sales`, `Quantity`, `Profit`) by centering the mean to `0` and scaling the variance to `1`.
    2.  **`PCA` (Principal Component Analysis):** Reduces dimensionality and extracts core variance components to eliminate potential multi-collinearity among financial metrics.
* **Categorical Data Path:**
    1.  **`OneHotEncoder`:** Transforms nominal text strings into sparse binary vectors, enabling the mathematical models to read categorical elements while safely ignoring unknown labels.

### Stage 2: Predictive Estimator (`XGBRegressor`)
The completely preprocessed and assembled feature array is passed directly into a trained **Extreme Gradient Boosting Regressor** (`XGBRegressor`).

* **Core Logic:** Gradient Boosted Decision Trees iteratively minimize prediction errors by building subsequent trees focused on structural residuals.
* **Hyperparameters:** Configured with 200 estimators and a maximum tree depth of 7 to accurately capture complex, non-linear correlations between structural profit margins, product sub-categories, and resulting optimal discounts.

---

## 4. Model Performance & Architecture

The final architecture is compiled into a single, highly transportable binary file:

* **`model.pkl`:** A completely self-contained serialized file housing both the trained preprocessing transformers (`StandardScaler`, `OneHotEncoder`, `PCA`) and the finalized `XGBRegressor` decision trees.
* **Implementation:** The pipeline can be loaded instantly into any Python data environment to make predictions on new data batches without requiring separate manual data encoding steps:
    ```python
    import joblib
    
    # Load the standalone model pipeline
    model = joblib.load('model.pkl')
    
    # Ready to predict optimal discount rates
    # predictions = model.predict(new_dataframe)
    ```
