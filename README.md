# car-accident-severity-prediction
## Car Accident Severity Prediction using Machine Learning

This project focuses on developing robust predictive models to classify the severity of car accidents, supporting business decision-making in the insurance industry. The goal is to assist a car insurance company in assessing risk, determining premiums, and improving customer safety services through data-driven insights.

## Problem Statement

Predict the severity of car accidents using historical data, with the target variable being `enhanced_accident`, which has multiple severity classes (multi-class classification). The dataset includes various features about the incident, driver, and environment.

## Models Used

- **Baseline**: Decision Tree Classifier
- **Advanced Models**:
  - Random Forest Classifier (with hyperparameter tuning via RandomizedSearchCV)
  - Gradient Boosting Classifier (with hyperparameter tuning)

##  Techniques and Tools

- Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn, Seaborn, Matplotlib)
- Class imbalance handling: **SMOTE**
- Model evaluation metrics:
  - Accuracy
  - Balanced Accuracy
  - AUC-ROC (multi-class)
  - Confusion Matrix

##  Key Findings

- **Random Forest** achieved high CV balanced accuracy (0.875) but **severely overfitted**, with test balanced accuracy dropping to 0.324.
- **Gradient Boosting** had a lower CV accuracy (0.674) but **generalized better**, with test balanced accuracy of 0.350.
- Both models showed **low discriminatory power** (AUC-ROC ~0.62), suggesting room for further improvement.

##  Future Improvements

- Implement stronger regularization on Random Forest
- Conduct feature engineering to capture interactions
- Test alternative models and ensembling techniques
- Consider integrating external data (e.g., weather, traffic)
- Address ethical concerns around predictive model use in insurance

##  Repository Structure
├── data/
│ ├── X_train.csv
│ ├── y_train.csv
│ ├── X_test.csv
│ └── y_test.csv
├── notebooks/
│ └── accident_severity_modeling.ipynb
├── results/
│ └── evaluation_plots/
├── README.md

##  Author

**Yogesh Dahiwal**  
Module: BNM872 - Machine Learning for Business Analytics  

---

##  References

- Leskovec, J., Rajaraman, A., & Ullman, J.D. (2020). *Mining of Massive Datasets*. Cambridge University Press.
- Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt Publishing.


