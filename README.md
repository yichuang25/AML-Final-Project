# AML-Final-Project

## Background and Context

Breast Cancer is the most common cancer among women in the world. It accounts for 25% of all cancer cases and affected over 2.1 million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area. The key challenge against its detection is how to classify tumors into malignant (cancerous) or benign (non-cancerous). 
The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mess. The data set[^fn1] can also be found in UCL Machine Learning Repository[^fn2].

## **Data Set Description**

The size of the data set is 569 rows with 32 columns. For each case, there is an ID number, diagnosis, and real-valued features computed for each cell nucleus. Diagnosis is categorical data which M means malignant, and B means benign. In this data set, there are 357 benign samples and 212 malignant samples.
The ten real-valued features are radius (means of distance from center to points on the perimeter), texture (standard deviation of gray-scale values), perimeter, area, smoothness (local variation in radius lengths), compactness (perimeter^2 / area -1.0), concavity (number of concave portions of the contour), symmetry, and fractal dimension (“coastline approximation” - 1). 
The mean, standard error, and “worst” or largest (mean of the three largest values) of these features were computed for each image, parts in 30 results. All feature values are recorded with four significant digits, and there are no missing attribute values in this data set.

## **Solution**

The critical challenge is how to classify tumors into malignant or benign. We plan to pre-process the data set, including one-hot encoding and finding the correlation between features. Because the number of benign and malignant samples is not even, so during the data split, we need to stratify the data during data separation for training, validation, and testing data set.
For machine learning techniques, we plan to use support vector machines (SVM), decision trees, random forests, and XGBoost to approach this classification problem. The biggest challenge of this project is the hyperparameter tuning and model validation. We plan to apply random search and grid search to find the best hyperparameter with k-fold cross-validation.

[^fn1]: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset 
[^fn2]:  [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) 
