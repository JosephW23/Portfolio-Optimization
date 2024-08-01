# Portfolio-Optimization

Links to our Milestones:

Milestone 2: https://colab.research.google.com/drive/19GPu_rviNHSLKBmy_38112gOe7dZswtD?usp=sharing <br>
Milestone 3: https://colab.research.google.com/drive/1Qv0hLxtsABuNxYEbuCjefbLN4LSL-cEM?usp=sharing

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

## Milestone 3: Pre-Processing

### Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

Question 1:

Linear Regression:The initial excellent R2 values of the Linear Regression model suggested that it was overfitting. This implies that it fits the training data—noise included—too closely, which may result in inadequate generalization to fresh data. This model would be on the right side of the fitting graph, where model complexity is high and overfitting causes the error on test data to start rising.

Ridge Regression:The appropriate range for model complexity is fit by the Ridge Regression model. By penalizing large coefficients, the regularization term helps prevent overfitting and achieve a balance between variance and bias. Good generalization is demonstrated by this model's low MSEs and excellent R2 scores on both test and training sets of data. Ridge Regression would be close to the bottom of the U-shaped curve in the fitting graph, where test and training errors are minimized and optimal model complexity is represented.

To compare the effects of regularization, we employed Ridge Regression and Linear Regression:A baseline for understanding the performance of a basic model in the absence of regularization was provided by linear regression. Its flawless R2 scores demonstrated that it assisted in identifying any overfitting problems.In order to solve the overfitting seen with Linear Regression, Ridge Regression was devised. Ridge Regression penalizes big coefficients, which lowers overfitting and enhances generalization to fresh data by including an L2 regularization factor. This illustrated how crucial regularization is to building a strong prediction model.

Question 2:

Investigating non-linear models, such as the Random Forest Regressor, can be very helpful for portfolio optimization in the following stages. This is the reason why: The Random Forest Regressor to enhance predictive performance, Random Forest is an ensemble learning technique that combines several decision trees. It records intricate interactions and non-linear correlations between features that may be overlooked by linear models. These kinds of associations are common in financial data, thus this can be quite helpful there.Advantages: By averaging several trees, it lessens overfitting and is resistant to noise and outliers in the data. It also offers feature importance metrics, which are helpful for comprehending the underlying causes of predictions, and manages both numerical and categorical characteristics with ease. Comparison with Linear Models: We can assess if incorporating non-linear interactions considerably increases the prediction accuracy and robustness of the portfolio optimization model by contrasting Random Forest's performance with that of the linear models (Linear and Ridge Regression).In situations when there are intricate relationships between the financial indicators, Random Forest may perform better, providing a possibly more accurate and trustworthy model for making decisions.

### Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

For both the training and test datasets, the Linear Regression model showed flawless R2 values, suggesting a significant level of overfitting. This flawless fit implies that the training data's noise and particular patterns, which are not very generalizable to fresh, unobserved data, were being captured by the model. Many approaches can be taken into consideration in order to enhance the Linear Regression model. Large coefficients can be penalized and overfitting can be decreased by using regularization techniques like Lasso Regression (L1 regularization) and Ridge Regression (L2 regularization). Furthermore, the model can be made simpler and more capable of generalization by limiting the number of features to just those that are most pertinent. Furthermore, overfitting can be lessened by evaluating the model's performance using cross-validation techniques and adjusting the hyperparameters accordingly.

In contrast to Linear Regression, however, the Ridge Regression model offered a more robust and balanced match. For the training and test datasets, it showed low MSE values and good R2 scores. The regularization term's addition reduced overfitting and produced a model that performs well when applied to new data. The robustness and dependability of the model were further validated by the cross-validation scores. Still, there may be room for advancement with the Ridge Regression model. Finding the ideal regularization parameter (alpha) value that reduces error and improves generalization can be accomplished by further fine-tuning it using methods like Grid Search or Random Search. Furthermore, enhancing or adding new features might help the model forecast more accurately by capturing more pertinent data.Exploring ensemble methods such as Random Forest or Gradient Boosting can capture more complex relatio

There are various ways to build on the Ridge Regression model's performance. Using models such as the Random Forest Regressor can aid in identifying non-linear patterns within the data. Model performance can be further improved by experimenting with different regularization strategies, such as ElasticNet, which combines L1 and L2 regularization. Robust and trustworthy predictions for portfolio optimization can be achieved by regularly assessing the performance of the model using cross-validation and modifying the modeling strategy in response to the findings. By following these procedures, we may create a more resilient model that fits the training set of data more accurately and generalizes to new data with greater efficacy, offering more trustworthy insights for portfolio optimization.
