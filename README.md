# Customer 360 Framework for Electronics Retail: RFM Segmentation, Churn Prediction, and Purchase Behavior Prediction 

* Role: Data Scientist 
* Tools: Python (pandas, scikit-learn, XGBoost, LightGBM, SHAP), Google Colab, Interactive Dashboard
* Dataset: E-commerce transaction logs — electronics retail store (Jan–Nov 2020, ~80K customers / 
https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-electronics-store )

## Business Problem
An electronics e-commerce store had rich transactional data but no unified view of who its customers were, who was about to leave, and how much they were worth going forward. Marketing and retention budgets were being spent uniformly across the customer base instead of being targeted at the segments where they would have the highest impact.

The goal of this project was to build a Customer 360 framework that answers three questions for every customer:
1. Who are they? (behavioral & value segmentation)
2. Will they churn? (predictive risk)
3. How much are they worth if retained? (predicted future revenue)

and to translate that into a prioritization framework the business could act on immediately.

## Approach & Methodology

### Data Cleaning & Preparation
* Cleaned raw event-level transaction logs (~millions of rows), handling corrupted timestamps, missing user_ids, zero/negative prices, and duplicate records.
* Engineered time-based features (hour, day-of-week, top-level product category) to support downstream analysis.
* Defined a strict time-window split to avoid data leakage: features built from a training window (Jan–Aug 2020), labels and revenue targets computed from a held-out observation window (Sep–Nov 2020).

### Exploratory Data Analysis
* Analyzed monthly order/revenue trends, category and brand performance, and customer purchase concentration (Pareto analysis of revenue contribution).
* Surfaced early signals of a highly skewed, single-purchase-heavy customer base — a key driver of the high baseline churn rate discovered later in modeling.

### RFM Segmentation
* Computed Recency, Frequency, Monetary (RFM) scores per customer and derived quintile-based segments: Champions, Loyal, Potential Loyal, At Risk, Lost.
* Champions and Loyal customers made up only ~34% of the base but generated 72.7% of total revenue ($68.37M), confirming a classic Pareto pattern and validating the need for segment-specific strategy.
 <img width="582" height="177" alt="image" src="https://github.com/user-attachments/assets/f93e5d3b-e05a-4427-8b0d-15125c22703b" />



### Churn Prediction
* Framed churn as: did the customer make zero purchases in the 3-month observation window?
* Built and benchmarked three models — Logistic Regression (baseline), XGBoost, and LightGBM — using stratified train/test splits and class-imbalance handling (scale_pos_weight / is_unbalance).
* Selected XGBoost as the best model based on ROC-AUC, then tuned the decision threshold (0.21) to optimize F1-score for business use.
* Used SHAP to explain predictions and validate that recency, frequency, and purchase-interval features were the dominant churn drivers — consistent with domain intuition.

### Model comparison
 <img width="771" height="126" alt="image" src="https://github.com/user-attachments/assets/04654f40-02c9-4e15-939b-8addbb9be010" />


### Customer Priority Matrix
Combined churn probability and predicted revenue into a 2x2 prioritization framework with four actionable segments
<img width="2148" height="711" alt="image" src="https://github.com/user-attachments/assets/feff5dd6-1304-4355-89e1-ca1243f75ee0" />

## Key Results
* Segmented 80,175 customers into 5 RFM tiers and 4 actionable priority groups.
* Identified that 72.7% of revenue comes from just 34% of customers (Champions + Loyal), sharpening retention focus.
* Built a churn model reaching 0.71 ROC-AUC / 0.81 F1-score, with an optimized decision threshold for real-world deployment.
* Forecasted $22.39M in predicted revenue for the following quarter, broken down by customer segment and priority.
* Translated model outputs into a prioritization framework directly usable by marketing (e.g., the "Grow & Upsell" segment — only 16% of customers — represents the highest average revenue and lowest churn risk, an ideal upsell target).

## Dashboard
https://datastudio.google.com/embed/reporting/492e5dd9-6654-4005-8503-a1f75c7ef0f8/page/JzE1F 
