# Supervised Learning for Real-World Problems 

### Introduction

In this task, I apply classic algorithms for classification and regression to predict the evolution of hospital patients' states and needs during their stay in the Intensive Care Unit (ICU). 

### Dataset
First step is handling specific data artifacts. Some common challenges we may face are **missing values**, an **imbalance of the labels**, or **heavy-tailed distributions** in the data.

The dataset contains the following files:

* train_features.csv:  **training set features**
* train_labels.csv:   **training set labels**
* test_features.csv:  **test set features**

The training (train_features.csv) and test set (test_features.csv) are both arranged in the same way. Each row contains the information for a patient at a given time identified by a unique patient ID and timestamp in columns pid and Time. The rest of the columns are medical information, either vital signs such as the Heart rate, or lab tests such as Calcium level in the patient blood. Finally, we are provided with the Age of the patient which is the same during the entire stay.

All medical measurements are not available at each timestep, meaning the data contains a lot of missing values, indicated with "nan" in our case. To simplify the problem, the data is already resampled hourly. This means that we aggregate measurements by one-hour period, thus there are only 12 rows for a given patient in the corresponding .csv file.

The last .csv file, namely train_labels.csv, contains the ground truth labels of the training set for each subtask. Each row is identified by a unique pid corresponding to a patient from the training set.

* For more details, check "Description.pdf".

### Tasks

11 classification tasks and 4 regression tasks based on the dataset.

* For more details, check "Description.pdf".

### Solution

#### Step1: Data Imputation

For data imputation, we have [6 different ways to compensate for missing values in a dataset](https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779). Considering the actual meaning of missing data, interpolation is the best way. It is much simple, and suitable for time-varying data imputing.

Implemented by ```pandas.DataFrame.interpolate()```

#### Step2: Binary Classification with probability / Regression

Generally, for [binary classification](https://machinelearningmastery.com/types-of-classification-in-machine-learning/), the classical algorithms are:

* Logistic Regression
* k-Nearest Neighbors
* Decision Trees
* Naive Bayes
* **Support Vector Machine**
* **Neural Network**

I solved this task by SVM classification and regression. 

My teammates Yue Li and Xue Ying solved it by Neural Network and Xgboost, respectively.

### Annotations

1. When processing data, "initialize a big numpy.ndarray" is better than calling "data.append()", to save time.

2.  The fit time of SVM scales at least quadratically with the number of samples and **may be impractical beyond tens of thousands of samples**. Unfortunately, our dataset size is 18995*420, which is too much for SVM. But luckily, the resulting hyperplane of SVM only depends on few support vectors, so **downsampling the train dataset**  to get a smaller dataset (around 10k) does cut down the time cost, but influences little on classification performance.

3. To deal with inbalanced data, we can set "class_weight" param to "balanced" in SVM. We also can downsample a balanced train subset from step 2.

4.  When regression, I apply ```train_test_split()``` to randomly downsample from train set. The regression results of 5% train set and 50% train set are compared as follows, which verified steps 2.

|percentage|1|2|3|4|
|----|----|----|----|----|
|5% samples| 18.18217446|   81.25058088| 97.52081263|82.1966722 |
|50% samples | 18.60417832|   81.86481371|  97.15250078 | 84.6593394 |

5. Other two solutions are provided as "solution_yue.ipynb" and "solution_ying.ipynb"

6. final score:0.752251889844
baseline:0.713853457215
current ranking: 59/199






