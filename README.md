# Customer 360 Framework for Electronics Retail: RFM Segmentation, Churn Prediction, and Purchase Behavior Prediction 

* Role: Data Scientist (End-to-End Project)
* Tools: Python (pandas, scikit-learn, XGBoost, LightGBM, SHAP), Google Colab, Interactive BI Dashboard
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
  
RFM Segment 	RFM Score	n Customer	%	Churn Rate	Interpretation
Champions	13 – 15	15.345	19,1%	61,0%	Active, frequent, high value
Loyal	10 – 12	21.179	26,4%	77,9%	Consistent engagement
Potential Loyal	7 – 9	21.500	26,8%	83,8%	Potential to be loyal
At Risk	5 – 6	13.778	17,2%	87,5%	Starting to be inactive
Lost	3 – 4	8.373	10,4%	88,0%	Long time inactive
Total	—	80.175	100%	78,9%	—
<img width="468" height="153" alt="image" src="https://github.com/user-attachments/assets/c768a4d7-64d8-4563-9ec5-4fc0eace1622" />

### Churn Prediction
* Framed churn as: did the customer make zero purchases in the 3-month observation window?
* Built and benchmarked three models — Logistic Regression (baseline), XGBoost, and LightGBM — using stratified train/test splits and class-imbalance handling (scale_pos_weight / is_unbalance).
* Selected XGBoost as the best model based on ROC-AUC, then tuned the decision threshold (0.21) to optimize F1-score for business use.
* Used SHAP to explain predictions and validate that recency, frequency, and purchase-interval features were the dominant churn drivers — consistent with domain intuition.

### Model comparison
Churn Risk	n Pelanggan	Persentase	Avg Churn Prob	Avg Monetary (USD)
Medium Risk (0,50–0,75)	55.340	69,0%	0,620	472,75
Low Risk (0,25–0,50)	20.656	25,8%	0,397	1.029,21
Safe (< 0,25)	4.179	5,2%	0,170	5.012,60
<img width="468" height="88" alt="image" src="https://github.com/user-attachments/assets/e11b02fc-8f5a-4fca-a961-dcd5e6af32ef" />

### Customer Priority Matrix
Combined churn probability and predicted revenue into a 2x2 prioritization framework with four actionable segments
