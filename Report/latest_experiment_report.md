# Experiment Summary Report (2026-10-30 21:17:00)

```text
================================================================================
EXPERIMENT RUN LOG — 2026-10-30 21:17:00
================================================================================

1. MASTER CONFIGURATION SETTINGS:
   - RANDOM_STATE           : 55
   - NUMBER_OF_FEATURES     : 4
   - RFE_SELECTOR_MODEL     : Logistic Regression
   - SFS_SELECTOR_MODEL     : Decision Tree
   - PCA_VARIANCE           : 0.95 (4 components kept)
   - CV_FOLDS               : 5
   - UNIQUE_FINALIST_MODELS : False

2. FEATURES SELECTED BY EACH METHOD:
   - Baseline (All 6)       : Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
   - ANOVA                  : Air temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
   - Pearson Correlation    : Torque [Nm], Tool wear [min], Air temperature [K], Rotational speed [rpm]
   - Adapted Lasso (L1)     : Torque [Nm], Rotational speed [rpm], Air temperature [K], Tool wear [min]
   - RFE (Logistic Regression) : Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm]
   - Forward SFS (Decision Tree) : Air temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
   - Backward SFS (Decision Tree): Air temperature [K], Process temperature [K], Torque [Nm], Tool wear [min]
   - PCA (95% var)        : 4 orthogonal components

3. BEST MODEL PER FEATURE-SELECTION METHOD (VALIDATION SCREENING):
   Feature selection Model  Validation failure F1  Validation failure recall  Validation failure precision  Validation missed failures  Validation false alarms
No feature selection   MLP               0.769231                   0.686275                      0.875000                          16                        5
                 RFE   MLP               0.702128                   0.647059                      0.767442                          18                       10
               ANOVA   MLP               0.674157                   0.588235                      0.789474                          21                        8
 Pearson correlation   MLP               0.674157                   0.588235                      0.789474                          21                        8
       Adapted Lasso   MLP               0.674157                   0.588235                      0.789474                          21                        8
   Forward selection   MLP               0.674157                   0.588235                      0.789474                          21                        8
  Backward selection   MLP               0.635294                   0.529412                      0.794118                          24                        7
                 PCA   MLP               0.500000                   0.352941                      0.857143                          33                        3

4. TOP 3 FINALISTS AFTER 5-FOLD CV HYPERPARAMETER TUNING (VALIDATION):
   Feature selection             Model  Validation F1  Validation recall  Validation precision  Validation missed failures  Validation false alarms
No feature selection Gradient Boosting       0.786517           0.686275              0.921053                          16                        3
No feature selection     Random Forest       0.777778           0.686275              0.897436                          16                        4
No feature selection               MLP       0.757895           0.705882              0.818182                          15                        8

5. STACKING ENSEMBLE Of Top 3 Fine Tuned Models(VALIDATION):
   Feature selection                  Model  Test F1  Test recall  Test precision  Test missed failures  Test false alarms
No feature selection Stacking (Tuned Top 3) 0.791209     0.705882             0.9                    15                  4

6. SELECTED WINNER & TUNED HYPERPARAMETERS:
   - Winning Model          : Gradient Boosting
   - Winning Feature Method : No feature selection (Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min])
   - Best Hyperparameters   : {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 1.0}

7. FINAL UNSEEN TEST SET PERFORMANCE (SINGLE WINNER VS. STACKING):
   Feature selection                  Model                                                                                                 Features  Test F1  Test recall  Test precision  Test balanced accuracy  Test missed failures  Test false alarms
No feature selection      Gradient Boosting Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] 0.813187     0.725490           0.925                0.861710                    14                  3
No feature selection Stacking (Tuned Top 3) Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] 0.791209     0.705882           0.900                0.851561                    15                  4
================================================================================

   - Test Failure F1        : 0.8132
   - Test Failure Recall    : 0.7255
   - Test Failure Precision : 0.9250
   - Test Balanced Accuracy : 0.8617
   - Test Missed Failures   : 14
   - Test False Alarms      : 3
================================================================================


```
