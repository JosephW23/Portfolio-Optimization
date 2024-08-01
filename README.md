# Portfolio-Optimization

Links to our Milestones:

Milestone 2: https://colab.research.google.com/drive/19GPu_rviNHSLKBmy_38112gOe7dZswtD?usp=sharing <br>
Milestone 3: https://colab.research.google.com/drive/1Qv0hLxtsABuNxYEbuCjefbLN4LSL-cEM?usp=sharing <br>

Link to Dataset: https://www.kaggle.com/datasets/macrosynergy/fixed-income-returns-and-macro-trends

## Milestone 2: Data Preprocessing. 

Missing Values Handling: The dataset was checked for missing values and discovered to be free of them. This is a good circumstance because it streamlines the preprocessing stages. However, it is critical to routinely check for missing values, especially after applying transformations or integrating additional datasets, to ensure data integrity is maintained throughout the research.

Remove Unnecessary Columns: The 'Unnamed: 0' column has been identified as an index column with no analytical value. Removing such redundant columns is critical since it reduces the complexity of the dataset and directs the analysis to more significant aspects.


Data Type Conversions: Data type consistency is critical for accurate analysis. The'real_date' column will be transformed to datetime format to aid in time series analysis. Furthermore, all numerical columns must be properly cast as float or integer types. This step helps to prevent type-related errors during analysis and modeling.

Handling Skewed Distributions: The dataset contains several numerical columns with skewed distributions, including 'value', 'eop_lag', and'mop_lag'. To reduce the skewness, a log transformation will be done. This transformation, which was already studied during the data exploration phase, aids in normalizing the data, improving the performance of many statistical and machine learning models that assume normally distributed data.

Scaling Numerical Features: Standardizing numerical features is critical for ensuring that all characteristics contribute equally during the modeling process. This will be accomplished with the 'StandardScaler', which scales the features to a mean of zero and a standard deviation of one. This step was partially handled during the log transformation of skewed columns and will be expanded to include other significant numerical aspects as well.

Handling Categorical variables, such as 'cid' and 'xcat', must be encoded in a numerical format in order to be employed successfully in machine learning methods. Depending on the modeling technique, one-hot or label encoding will be used. One-hot encoding is often chosen for categorical variables without an ordinal relationship because it prevents the algorithm from assuming an inherent order in the categories.

Feature: Feature entails adding new features from current data to improve the model's prediction potential. Time-based elements like year, month, and quarter will be derived from the'real_date' column to capture seasonal trends and patterns. If the variables are considered to have non-linear correlations, interaction terms or polynomial characteristics can be derived.

Outlier Detection and Treatment: Outliers can have a major impact on model performance, particularly for those that are sensitive to data scale. Outliers will be identified and treated using techniques such as the Interquartile Range (IQR), Z-scores, and more complex methods such as isolation forests. Depending on their impact, outliers can be deleted or changed to reduce their influence.

Splitting the Dataset: To evaluate a model, it is necessary to measure performance on previously unseen data. As a result, the dataset will be divided into training and testing sets using 'train_test_split'. A common split ratio is 70-80% training, 20-30% testing. This split allows the model to be trained on a large chunk of the data while being tested on a distinct sample to assess its generalization abilities.

