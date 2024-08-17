    # Comparison of Machine Learning Algorithms for Water Quality Prediction

This repository consisted data set and files used for this project

## Table of Content

- [Introduction](#Introduction )
- [Objectives](#Objectives)
- [Materials and Methods](#Materials-and-Methods)
- [Description of Data Set](#Description-of-Data-Set)
- [Software Used](#Software-Used)
- [Methodology](#Methodology)
  - [Data Acquisition](##Data-Acquisition)
  - [Data Preprocessing](#Data-Preprocessing)
  - [Missing Value Identification](#Missing-Value-Identification)
  - [Filling Missing Values](#Filling-Missing-Values)
  - [Normalization](#Normalization)
  - [Outlier Identification and Removal](#Outlier-Identification-and-Removal)
- [Train and Test Method](#Train-and-Test-Method)
- [Model Training](#Model-Traning)
- [Model Evaluation](#Model-Evaluation)
- [Result and Discussion](#Result-andDiscussion)

## Introduction 

Water covers approximately 70% of the Earth's surface and is essential for sustaining life. However, rapid urbanization and industrialization have significantly deteriorated water quality, leading to an increase in waterborne diseases. Traditionally, water quality has been assessed through costly and time-intensive laboratory tests and statistical methods, which challenge the practicality of real-time monitoring. Various methods, such as multivariate statistical techniques, fuzzy inference systems, and Water Quality Index (WQI) calculations, have been employed to predict water potability. These approaches often involve monitoring numerous water quality parameters according to established standards, with the final evaluation results varying widely depending on the parameters chosen.

In recent years, advancements in machine learning have revolutionized water quality assessment. Machine learning techniques enable the capture and analysis of vast amounts of data, addressing the complexity and scale of modern water quality evaluation needs. As a result, these methods offer the potential for more accurate and reliable water quality assessments.

## Objectives

This project aims to identify the most accurate and reliable machine learning algorithm for predicting water quality by comparing various ML models. The project focuses on:

* Familiarizing with data cleaning and Exploratory Data Analysis (EDA).
* Understanding and use of model construction 
* Understanding and applying model evaluation metrics to assess the accuracy of different models.
* Understanding evaluating the performance of these models on validation and test datasets.
* Comparison of machine learning algorithms to predict water quality.
* Determining the best model for predicting drinkable water quality.

## Materials and Methods

### Description of Data Set

The datasets employed in this project contains 10 water quality metrics for 3276 different water bodies. The dataset can be downloaded freely from the [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability). The water parameters that are available in the dataset are listed following table

| Parameter| Variable Definition | Min | Max|
| -------- | -------- | -------- | -------- |
| pH |Acid–base balance of water (0-14)| 1 | 14 |
| Hardness | Capacity of water to precipitate soap in mg/L | 47.432 | 323.124 |
| Solid| Total dissolved solids in ppm | 320.9426	| 61227.2 |
| Chloramines | Amount of Chloramines in ppm | 0.352 | 13.127 |
| Sulphate | Amount of Sulphates dissolved in mg/L | 129 | 481.0306 |
| Conductivity | Electrical conductivity of water in μS/cm | 181.4838 | 753.3426 |
| Organic Carbon | Amount of organic carbon in ppm | 2.2 | 28.3 |
| Trihalomethanes | Amount of Trihalomethanes in μg/L | 0.738 | 124 |
| Turbidity | Measure of light emitting property of water in NTU  | 1.45 | 6.739 |
| Portability | Potable - 1 and Not potable - 0  | 0 | 1 |

### Software Used

This project was completed using [Orange](https://orangedatamining.com/) version 3.34.0, an open-source platform for machine learning and data visualization. Orange provides a user-friendly interface to build data analysis workflows visually, offering a large and diverse toolbox to suit various analytical needs. Following shows some of key features in the software.

* **Open Source Machine Learning and Data Visualization**: Orange is a powerful open-source tool that supports machine learning and data visualization, making it accessible and customizable for various research and analysis tasks.
* **Visual Workflow Construction**: The project leveraged Orange’s visual programming capabilities, allowing for the creation of complex data analysis workflows by simply dragging and dropping widgets. This approach makes it easy to understand the process flow and modify the analysis as needed.
* **Diverse Toolbox**: Orange offers a rich set of widgets for data preprocessing, visualization, and modelling, enabling comprehensive analysis. In this project, several of these tools were utilized to explore, preprocess, and model the data effectively.

## Methodology 

Following shows the proposed methodology for predict water quality using different ML algorithms.

![Methodology](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/methodology.png)

Hence this project is done by Orange software, above proposed methodology is utilized. Following shows the visual workflow construction done by Orange according to the proposed methodology.

![workfolw](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/orange%20work%20flow.PNG)

following describe all the steps involved in this project using Orange software

### 1.	Data Acquisition

Data set was downloaded as .csv format. There are 3276 data with 10 Parameters.  

a) Once data downloaded csv file was imported in to Orange. 
open *file widget* in orange interface ---> right click ---> Cick open ---> Browse data set 

![Data access](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/data%20acuqisition.png)

b) Next open **Data Table widget** receives one or more datasets in its input and presents them as a spreadsheet. Data instances may be sorted by attribute values. It shows all numeric and null values as a tabular format.

![Data](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/data%20acuqisition%202.png)

### 2.	Data Preprocessing 

Preprocessing helps transform data so that a better machine learning model can be built, providing higher accuracy. The preprocessing performs various functions: outlier rejection, filling missing values, data normalization to improve the quality of data. 

**a) Missing Value Identification**

Then open **feature statistic widget** to inspect and find potentially interesting features in the given data set. Histogram showing the distribution of feature's values. Further columns show different statistics. Mean, median, missing minimal and maximal value are computed only for numeric features. When consider about the missing values in data set, following missing values (percentages) for each parameters were identified. 

![Missing value identification](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/missing%20value%20identification.png)

| Parameter |Number of missing values |
|----------|----------|
| pH| 491 (15%)|
| Hardeness| 0 |
| Soild| 0|
| Chloramines| 0|
| Sulphate| 781 (24%)|
|Conductivity| 0|
|Organic Carbon| 0|
|Trihalomethanes|162 (5%)|
|Turbidity|0|
|Portability| 0|

**b) Filling Missing Values**
Then missing values were replaced the missing value with the corresponding mean value using **impute widjet.**

![filling missing values](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/filling%20missing%20values.png)

**c) Normalization** 

The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information. Performed feature scaling by normalizing the data from 1 to -1 range (z-score normalization) using **continiuze widget**.

![Normalization](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/normalization.png)

Once normalization finished, the new data table is used for further processing.
* **Save data widget** – to save new table in to local folder
* **Feature statistics widget** – to check missing values again. After cleaning the table, it is identified that no missing values for each parameters
* **Scatter plot widget** – The Scatter Plot widget provides a 2-dimensional scatter plot visualization between two parameters

Following shows the scatter plot between two paramenters after doing normalization

![normalization result](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/result%20normalization.png)

**d) Outlier Identification and Removal**

Outliers are the data points that are significantly different from the rest of the dataset. Using the **outliers widget** , filtered the dataset for detecting outliers and inliers. The number of outliers in the data set is 3000 and inliers are 276.

![outlier](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/outlier%20identification%20and%20removal.png)

### 3.	Train and Test Method

After data cleaning and preprocessing, the dataset becomes ready to train and test. Using **data sampler widget**, **75%** train/test splitting method is used to test the different machine learning model’s Performance separately. In the train/split method, split the dataset randomly into the training and testing set.
* Number of data in training set – 2250
* Number of data in test set - 750 
* Total – 3000

![split](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/train%20and%20test%20method.png)

### 4.	Model Training 

In this project, comprehensive studies are done applying different ML techniques like decision tree, random forest, logistic regression, k-nearest neighbours, support vector machine. 

**Decision Tree**

A decision tree is a simple self-explanatory algorithm, which can be used for both classification and regression. After training, the decision tree makes predictions by evaluating the values of relevant input features. It typically uses a metric like entropy or Gini impurity to select the best feature to split on at each node, starting with the root. The tree is structured in a top-down manner, where each internal node represents a decision based on a particular feature, and each branch represents an outcome of that decision. The process continues until the algorithm reaches a leaf node, which provides the final prediction based on the values of the input features. In this project **tree widget** is used for build decision tree model.

![decision tree](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/decision%20tree.png)

**Random Forest**

Random Forest is an ensemble learning method used for both classification and regression tasks. It operates by constructing a multitude of decision trees during training and outputting either the mode of the classes (in classification) or the mean prediction (in regression) of the individual trees. For each tree, a random subset of features is considered when making splits, which introduces more diversity among the trees. In this project **random forest widget** is used for build random forest model.

![random forest](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/Random_forest_explain.png)

**Logistic Regression**

Logistic Regression is a supervised learning algorithm used to predict the probability of a binary outcome (e.g., yes/no, true/false, 0/1).The model is based on the logistic function, also known as the sigmoid function, which maps any real-valued number into a value between 0 and 1, representing the probability of the event occurring.Logistic regression is typically employed to predict a categorical dependent variable (the outcome) using one or more independent variables (features). These independent variables can be either continuous or categorical. In this project **logistic regression widget** is used for build logistic regression model.

![logistic regression](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/logistic%20regression.png)

**K-Nearest Neighbors**

K-Nearest Neighbors (kNN) is a simple yet effective classification algorithm that classifies a data point by identifying its closest k neighbors in the training dataset and assigning the most common class (majority vote) among these neighbors.kNN is often not recommended for large datasets because it is computationally intensive during the prediction phase. For each new data point, the algorithm must calculate the distance to all points in the training set to determine the nearest neighbors, which can be slow and resource-demanding as the size of the dataset increases. In this project **knn widget** is used build for K-Nearest Neighbors model.

![knn](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/knn.png)

**Support Vector Machine**

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for both classification and regression tasks, though it is primarily used for classification.The main objective of the SVM algorithm is to find the optimal hyperplane that best separates the data points into different classes in an N-dimensional space, where N is the number of features. The hyperplane is chosen to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class (known as support vectors). A larger margin leads to better generalization on unseen data. SVM is particularly effective in high-dimensional spaces and is versatile due to its ability to use different kernel functions (e.g., linear, polynomial, radial basis function) to handle non-linearly separable data. In this project **SVM  widget** is used for build Support vector machine model.

![SVM](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/support-vector-machine-algorithm.png)

Following shows the work flow to use testing and training data to train each models.

![model train](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/modls.png)

### 5.	Model Evaluation

Model evaluation is the process of using different evaluation metrics to understand a machine learning model's performance, as well as its strengths and weaknesses. Model evaluation is important to assess the efficacy of a model during initial research phases, and it also plays a role in model monitoring. following methods are used for model evaluation stage.

* Cross Validation
* Performance parameters 
  * Classification accuracy
  * Training time and Test time
  * Area under curve (AUC)
  * Precision
  * Recall
  * F1 Score

**Test and core widget** is used for Evaluation Results, results of testing classification algorithms.

**Cross Validation**

The data set was split into two parts: training (75%) and testing (25%). The training set was subjected to repeated cross-validation, with the number of iterations fixed to Classifiers were trained in this manner. The model's optimal parameter configuration was selected, resulting in the maximum accuracy. In this project number of folds for cross validation is 5.

![CV](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/grid_search_cross_validation.png)

**Performance Parameters**

**Classification Accuracy**

The accuracy of the machine learning algorithm can be calculated from the confusion matrix

The accuracy is given by:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

where:
- $TP$ is True Positives,
- $TN$ is True Negatives,
- $FP$ is False Positives, and
- $FN$ is False Negatives.

**Training time and Test time**

Training time refers to the time when an algorithm is learning a model from training data. Test time refers to the time when an algorithm is applying a learned model to make predictions.

**Area under curve (AUC)**

AUC measures the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1). When AUC is increase the performace of the model is also increse.

**Precision**

Precision is one indicator of a machine learning model's performance – the quality of a positive prediction made by the model. Precision refers to the number of true positives divided by the total number of positive predictions

The precision is given by:

$$
\text{Precisiony} = \frac{TP}{TP +FP}
$$

where:
- $TP$ is True Positives,
- $FP$ is False Positives

**Recall**

The recall measures the model's ability to detect Positive samples. Recall (or True Positive Rate) is calculated by dividing the true positives by anything that should have been predicted as positive.

The recall is given by:

$$
\text{Precision} = \frac{TP}{TP +FN}
$$

where:
- $TP$ is True Positives,
- $FN$ is False Negatives

**F1 Score**

F1 score is the performance measure over testing accuracy. It actually indicates that how stable the model is to predict the classes. If the F1 score is higher than the testing accuracy, then the system is more stable and accurate according to recall.

The F1 score is given by:

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

## Result and Discussion

Followinng table shows the overall results given by each evaluation tests.

| Parameter  | Logistic Regression | Decision Tree | Random Forest | KNN  | SVM  |
|------------|---------------------|---------------|---------------|------|------|
| **AUC**    | 0.542               | 0.991         | 0.841         | 0.66 | 0.525|
| **CA**     | 0.614               | 0.991         | 0.764         | 0.657| 0.52 |
| **F1**     | 0.492               | 0.991         | 0.755         | 0.642| 0.526|
| **Precision** | 0.549            | 0.991         | 0.764         | 0.644| 0.536|
| **Recall** | 0.614               | 0.991         | 0.764         | 0.657| 0.52 |

**Train and Test Time**

When consider about the train and test time for each algorithms, **KNN** has quick training time and **decision tree** has quick testing time.

| Algorithm            | Train | Test  |
|----------------------|-------|-------|
| **Logistic Regression** | 0.217 | 0.039 |
| **Tree**             | 0.482 | 0.001 |
| **Random Forest**    | 1.943 | 0.076 |
| **KNN**              | 0.153 | 0.509 |
| **SVM**              | 1.766 | 0.31  |

![time](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/time.png)

**Confusion Matrix**

The diagonal are examined and the values of the confusion matrix belonging to each decision tree method are compared to each other, Diagonal numbers indicate the number of accurately predicted data. It is obvious that the maximum number is in **decision tree** Model.

![CM](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/CM.png)

**AUC Curve**

The AUC value is highest for **decision tree** Model and then **random forest**.

![AUC](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/auc.png)

When consider about the precision, recall and f1 score values **decision tree** got the highest value then **random forest**

![overall](https://github.com/UpekshaIndeewari/Comparison-of-Machine-Learning-Algorithms-for-Water-Quality-Prediction/blob/main/images/overall.png)

As a result,
When comparing the performance parameters for each models Accuracy, precision, F1 score, AUC and recall gives higher values for **Decision Tree Model** and **Random forest** model than other models. It is understood from this study that it is possible to predict water quality with Decision tree model and Random forest model which close to reality.




























