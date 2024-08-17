# Comparison of Machine Learning Algorithms for Water Quality Prediction

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

The datasets employed in this project contains water quality metrics for 3276 different water bodies. The dataset can be downloaded freely from the [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability). The water parameters that are available in the dataset are listed following table

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

following describe all the steps involved in this project using Orange software

### 1.	Data Acquisition

Data set was downloaded as .csv format. There are 3276 data with 10 Parameters.  

a. Once data downloaded csv file was imported in to Orange. 
open *file widget* in orange interface ---> right click ---> Cick open ---> Browse data set

b. Next open Data Table widget receives one or more datasets in its input and presents them as a spreadsheet. Data instances may be sorted by attribute values. It shows all numeric and null values as a tabular format.

### 2.	Data Preprocessing 

Preprocessing helps transform data so that a better machine learning model can be built, providing higher accuracy. The preprocessing performs various functions: outlier rejection, filling missing values, data normalization to improve the quality of data. 

**a. Missing value identification**

Then open feature statistic widget to inspect and find potentially interesting features in the given data set. Histogram showing the distribution of feature's values. Further columns show different statistics. Mean, median, missing minimal and maximal value are computed only for numeric features. When consider about the missing values in data set, following missing values (percentages) for each parameters were identified. 

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













ee 
