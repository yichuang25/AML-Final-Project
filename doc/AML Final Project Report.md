<h1>
  AML Final Project Report
</h1>

# Introduction

Breast cancer is the most common cancer among women in the world. It accounts for 25% of all cancer cases and affected over 2.1 Million people in 2015 alone. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

In this project, we will apply machine learning models to help classify benign or malicious tumors through the dataset from Breast Cancer Wisconsin (Diagnostic)[^f1]. We will use logistic regression, support vector machine, decision tree, and boosting techniques to approach the classification problem.

[^f1]:  https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset 

# Data Preprocessing

The data set contains 30 features after dropping meaningless features such as the patient's ID in the data exploration process. To eliminate the multicollinearity of the features, we computed the correlation matrix and removed the features that caused more than 0.95 correlations. During data exploration, we also noticed that the data set was imbalanced. There are 62.7% benign samples and 37.3% malicious samples. We will apply to stratify technique during the train test split process and adopt the f-1 score as the key metric. We also employed sampling methods such as oversampling or SMOTE to increase the number of samples.

# Machine Learning Models

## Logistic Regression

After dropping the highly correlated features, logistic regression is a desirable and simple machine learning model to apply to predict categorical targets, which are benign and malicious tumors. In this section, we design and implement simple logistic regression and logistic regression with regularization, such as L1, L2, and both. We applied a grid search with cross-validation to discover the best hyperparameter for each model. Because the dataset is imbalanced, the f1 score is a good metric for cross-validation. The performance of each model on test data is below:

|             | Accuracy | Precision | Recall | F1 Score | AUC    |
| :---------- | -------- | --------- | ------ | -------- | ------ |
| Simple      | 0.947    | 0.974     | 0.881  | 0.925    | 0.9511 |
| Lasso       | 0.982    | 1         | 0.952  | 0.976    | 0.9980 |
| Ridge       | 0.982    | 1         | 0.952  | 0.976    | 0.9977 |
| Elastic Net | 0.982    | 1         | 0.952  | 0.976    | 0.9960 |

From the table above, confusion matrix and ROC plot, regularization helps improve the model from all metrics. Compared with the model with regularization, Lasso regression has a higher AUC score. The top 5 weighted features in the lasso regression model are area_worst, concave points_mean, texture_worst, concave points_worst, and symmetry_worst.

## Support Vector Machine (SVM)

For support vector machines, we explored primal and dual SVMs to understand which model would perform better on our dataset. The accuracy of baseline Primal SVM and Dual SVM is 0.964 and 0.984. Both models perform well, and Dual SVM has a greater f1 score. And after hyperparameter tuning through grid search, the accuracy of the Dual SVM model changed to 0.974. Considering the accuracy of the test dataset was quite high, it seems that choosing to use SVMs has also helped the model with overfitting. 

For both models, texture_worst was the most important feature by SHAP value, with the primal SVM having “conacve_points_mean” and “concavity_worst” as the second and third, respectively, while the dual SVM had “concavity_mean” and “concave_points_mean” as its second and third by SHAP value. 

## Decision Trees

Our investigation of tree-based models started with decision trees and expanded to balanced decision trees. Since the dataset was slightly skewed in the number of classes represented, we applied random oversampling to upsample the minority class, as there were not enough data points to give a great score estimate. We noticed a performance improvement by doing this and applying the balanced decision tree with a high average precision score of 0.92 compared to 0.82, with the f1 score being 0.94. When exploring SHAP values, the features that contributed most to the results were ‘concave points_mean’, ‘concave points_worst’, and ‘area_se.’

## Boosting

We then applied and compared models such as Random Forest, XG boost, and Histogram gradient boosting. The following models perform well for binary classification problems such as ours, which is to classify a benign or malignant cancer. We tune our hyper-parameters for each of these models by using Grid Search CV and varying different impactful hyper-parameters. In order to evaluate our models, we predict the metric score and output the precision, recall, and F1 score. Finally, we use the ROC curve to compare the different models. The accuracy for each model is very high, and the precision, recall, and F1 score are also excellent. The histogram gradient boosting is one of the greatest, with accuracy, precision, recall, and F1 score of around 97 percent and the shortest time to compute (6 seconds). We added its confusion matrix below. In addition, we calculated the feature importance using the SHAP library. The importance of the features differs in their measure across the different models; however, the top ones remain the “area worst,” “concave points mean, “ and “concave points worst”.

# Conclusion

In conclusion, we have high accuracy all around, which may be due to the nature of our dataset. Our methods have also minimized overfitting, with all models performing well on both the test and the training dataset. Additionally, we know from the SHAP analysis of each model that the features marked as most important changed based on the model, which indicates that the nature of the model did indeed change the outcome. 

XGboost produces the best model (auc=0.9964), but several models with accuracies were in the same range. It is our belief that fairly straightforward, and multiple models could be put into products with comparable results.

# Appendix I

## Performance of Logistic Regression

![img](assets/9b4DvhDWF9ZOKjQU_bW9b8-I7wQKL0KUhHGC_xZ1GOMNS0wp_cs91hWgTa98w_4KM3007DjiKBerIVg1Vx40fqHwa9B60-DopzkcEy0C2Czdvukujz1BxDv_3Mxbw4BothUrk5XZBJYSLHMbvSc6AqiNL2z-fExCp540D9FEH_RN-J1AclRxHqTtT-pHOQ.png)

![img](assets/c5XGZxy3LGov4Esu-pm7_BaxE9aYLE5FMCvjDaM9B6tFvxzRLUIoWn1TMNfkX7E0zFCJ0T66R68oj6X54AV5RukmyNvi5juMbrm9wG1HDQyx4Ag8-atDS8_K1Hp3wZpxLpUsQp4TuhZciFCfsYles5QF1Cv9BNVqB8yfKNeKYBfuFgbiQctmU9KXj4nQLg.png)

![img](assets/u2KDPU-LRxDiOXxG9VUUqnC2JRijL-T0FBfPVXwG8eurd2LNTxnvpKqHIzQoUrLXKeG6gPT2XYU0NQG9VPcnNsn3DVpkjUtaeyuURr9oCdzQ52sR1wujXdbkG195nzfIQXH-o2G5NSVG70jj6qnc4T7bDhxcGd2LHUThQlerrL29tCRYaQv1O6hqazuSkw.png)

## Performance of SVM

![img](assets/MdLt9e3YvNxzrxrjwL4DoySTs2EI1lhMusxvufV26m6dzSl8xuL1s3LXYh7bryUjcpjj1Tq6bwaInkJIQP8RbAnZfNSU5ydVJoAlwWtvA5xbyWaU3EPWpYkrNkWFujdCF-89J6jnBDa696lYWB_w6Eezm1xnjwBQ5ISh4nxlXRYeBE3n_KZzox27nLAe2w.png)

![img](assets/JF6ocA2k24v-QKfPMDsvQipAaXynjAZEnGDewM8cYUlAyEFEnMQ7jLJMl1iPBemsWUnX2vwxuZu4s5XQs1RdWywxVaUyN8UcmtBAkw8n0dgx7mCKifRLiBNiWS6kYph--ul2DGWMRwHUiF_f9MKoLCwZLSK70WDV-OM9eVT8X3_f0VNXc7Omf6e4--25XQ.png)

![img](assets/TSwAMLIu5foCduPM8rSn0DdGxds5v1yt-ZSPm4fT_DgIoGuBZhLCkoI4AaCseJQb9pek5xupy20Ndk2vs_62p4crlnYcfJVUXPG3AWGOpB2O9diCN6LYfmbLQ3nQTXE1n_wryJuexfIN9XaTllotbZALFVs_wC-QiPL47NZKZqoZN_e-iUTy4IjTifUTvQ.png)

## Performance of Balanced Decision Tree with Oversampling

![img](assets/E5oMH9_vafHd8dbJAvBFj2x0w15-or-gb1duzUx6QI82Y-qFecZy2LUrN0Ex-2LbXThUKzZxmEo8q4g4WGrz9_anN_Pp3X8_Dq8Aen7naBYiuv_G0NHZt5tRxyDvW0Dh8hQf367lXy7Iqvv2NihhLksFyZ-ECHBBX4MneJVOv-UjK7Is4hOvdNNkCobYHw.png)

![img](assets/BM49abi576zBqZTBebd3BNOsDSaYX9yN0x5JzjxFrLvnQVmPMOHXnXIvPzzGaSK9SFWNKhILDD96-pDeuXLseMKQ46Clyck9VG1mLZc-OlqNsIzY15nVV_AuJFKfJzGmx56NaHrYB99j_GkggCP0fMowDkyxeMptIkw7MIsOyXHktsP2TZP0ODEYRd9Aiw.png)

## Performance of Boosting

![img](assets/SiYa09COlSIhDxl1uf9_RSyC62SHJ4kkh2SGqbBauxX3d4JPYM4FTbxiVa8d0uFs4_5t7J82b2SwNw42QE3vkqz2lPYl2V6_8uy6l91_qUIeSqZQTlwgXK2jts1LV-jGoVOcD-kSUfW7w7HQWOgnlxshT6Ns2HYladPQhUHAHO6knQ0_VDuCjFeWhxWakg.png)

![img](assets/nArYZRKL0VVIFOIaHSqJ_SWJHmvnPi0d99SPTa0NBazBwOHnoC2h1_1EnaGTaHAkquuvmvQ3fTQQXG5VLwaKq-kZ3HH-gNp__J7UyaOlGFv_DQalsHmiDmLu_mZ--QtSiiYT-2h0J_bjHbqVMfuxeOWVejk1s49MjL3H9-MMUTx1yhL6HF9bUqSXfIr5dw.png)

![img](assets/3zgilc6yLhU2oKs70GGQ9d_G0fKUaY0NhvrGsXO1jR5R6CiNaZDi9A6UH6SVmFopWDdc0uAoNrrxJLpakq4p4qR2xylSq7WwJ_DbOA52Rr7p7vrMBQq2WRMpFxx6as7hrkZ_oZ_PqsjFz46FWXdzeskqFmDj04u1CPE6BfdPMX4A_o7oqznGDBWfopr3vw.png)