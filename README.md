### Background Information: 
Dean De Cock (a professor at Truman State University) compiled the "Ames Iowa: Alternative to the Boston Housing Data Set."  He found this data from the Ames Assessor’s Office (used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010), and he used the project for a semester project for an undergraduate course in regression.  According to Dr. De Cock, the original data is used for tax assessment purposes but can also be used to predict home selling prices because the type of information contained in the data is similar to what a typical home buyer would want to know before making a purchase.

### Problem Statement/Goal: 
The goal of the project is to use Ames, Iowa housing data and linear regression modeling techniques to predict property prices (in Ames, Iowa during 2006 to 2010).  The goal is to also determine which features increase house sale price and which decrease house sale price.  A linear regression model to predict house sale price will be created, and its accuracy and precision in prediction will be assessed. 

### Project Outline
1. Import Ames, Iowa housing dataset
2. Clean the data (with a focus on null values)
3. Perform Exploratory Data Analysis (EDA) to find relationships between features and the target variable (SalePrice) and to understand whether assumptions required to build a multiple linear regression (MLR) are met.
4. Create linear regression models, and choose one that is highly accurate and precise (using MSE, RMSE, and MAE values as well as R squared score values) to predict sale price
5. Analyze model parameters to see what the most important features are, and present findings to stakeholders (such as a real estate company, home owners, and/or home buyers)

### Data Dictionary
See the data dictionary here: http://jse.amstat.org/v19n3/decock/DataDocumentation.txt (author: Dean De Cock)

One feature was created through addition of four other features (a feature interaction term built through addition rather than multiplication):

|Feature|Type|Dataset|Description|
|---|---|---|---|
|Total bathrooms|*float*|Created from Ames, Iowa housing dataset|Basement Full Baths + Basement ½ Baths + 
Above Grade Full Baths + Above Grade ½ Baths|

### Repository Structure

#### Folders within project_2 Folder: 
- datasets 
- Julia Taussig - DEN
- suggestions (markdown file from General Assembly)

#### Contents of datasets folder:
- sample_sub_reg.csv: an example of a submission to Kaggle from General Assembly (shows format required for submission)
- test.csv: the test data set from General Assembly (used for the Kaggle competition) 
- train.csv: the train data set from General Assembly

#### Contents of Julia Taussig - DEN folder:
- code folder
- data folder
- images folder
- Proj_2_PresnJuliaTaussig.pdf: presentation to stakeholders including recommendations using insights from model

#### Contents of code folder:
- Jupyter Notebook: 1_EDA_and_Cleaning.ipynb
- Jupyter Notebook: 2_Preprocessing_Dummies_Analysis_BaselineModel.ipynb
- Jupyter Notebook: 3_FeatureEng_ModelTuning_KaggleSubmissions_Metrics_Visualizations.ipynb

#### Contents of data frolder:
- test_clean_1.csv: the partially clean test data set from General Assembly and Kaggle 
- test_clean_2.csv: the clean test data set from General Assembly and Kaggle (null values were assessed and resolved)
- test_df_dum.csv: the cleaned test data set (for the Kaggle competition) with dummy variable columns appended
- train_clean_1.csv: the partially clean train data set 
- train_clean_2.csv: the clean train data set (null values were assessed and resolved)
- train_df_dum.csv: the cleaned train data set with dummy variable columns appended
- submission folder

#### Contents of submission folder: (submitted SalePrice prediction results with corresponding ID's submitted to Kaggle for competion)
- submission_lr_1.csv: Baseline model created using linear regression of all features (but forgot to inverse-transform the predictions)
- submission_lr_model_1_InvTrans.csv: Baseline model created using linear regression of all features with inverse-transformed predictions
- submission_lasso_model_2.csv: Model created using linear regression with LassoCV regularization (but forgot to inverse-transform the predictions) 
- submission_lasso_InvTrans_model_2.csv: Model created using linear regression with LassoCV regularization with inverse-transformed predictions
- note that Model 3 was a model created using linear regression with RidgeCV regularization with inverse-transformed predictions, but it was not submitted due to poor R squared scores.
- submission_enet_InvTrans_model_4.csv: Model created using linear regression with ElasticNetCV regularization with inverse-transformed predictions 
- submission_ridge_InvTrans_model_5.csv: Model created using linear regression with RidgeCV regularization with inverse-transformed predictions (included features with correlation with SalePrice > 0.4)
- submission_lasso_InvTrans_model_6.csv: Model created using linear regression with LassoCV regularization with inverse-transformed predictions (included features with correlation with SalePrice > 0.4 as well as interaction feature)
- submission_ridge_InvTrans_model_7.csv: Model created using linear regression with RidgeCV regularization with inverse-transformed predictions (included features with correlation with SalePrice > 0.3)
- submission_ridge_InvTrans_model_8.csv: Model created using linear regression with RidgeCV regularization with inverse-transformed predictions (included features with correlation with SalePrice > 0.4 and increased percent of train data used for training from 70% to 75% (decreased train data used for testing from 30% to 25%)

### Executive Summary
#### Data Importing and Early EDA and Cleaning
After importing the train and test datasets, data was inspected. Both the train and test dataset had many null values, and these values were inspected using the data dictionary and context about the properties' rows.  The missingno library's matrix was used to visualize null values density in train and test dataframe columns.  Feature distributions were inspected using boxplots and scatterplots.  A function was used to list features with outliers.  The distribution of SalePrice was also inspected.  There was a right-skew (positive skew) in the SalePrice distribution, and there were a significant amount of outliers in the SalePrice and feature data, so a PowerTransform was considered to be used for modeling. A heatmap of correlation of features to SalePrice was used to visualize which features initially correlate with SalePrice (before cleaning).  

Data was then cleaned starting with inspection of variables with the least null values.  There was a row in the training set with null values for Garage Yr Blt, Garage Finish, Garage Cars, Garage Area, Garage Qual, and Garage Cond even though it had a Garage Type of Detached.  This means that there were several typos in the row.  This row was therefore removed from the dataframe because it was only one property of thousands of properties, and it had obvious errors that could harm model prediction accuracy and precision.  There was only one property in the training set (PID: 528142130 and index 1147) with a basement (instead of with no basement as depicted by an 'NA' value in the Bsmt Cond column) that had a null value in the BsmtFin Type 2 column. The Basement quality was good, the basement condition was typical, and the basement finish type 1 was GLQ (good living quarters). Other properties with good basement quality, typical basement condition, GLQ basement finish type 1, and BsmtFin SF 2 values greater than 0 had BsmtFin Type 2 values of Rec, ALQ, LwQ, BLQ, or Unf. It was hard to predict what the value should be for this property's BsmtFin Type 2, so this row was removed.  There were several null values that were replaced with "NA" or 0 or "No" or "None" because the context of similar variable values in the row proved these would be appropriate replacements.  For example, when Total Bsmt SF = 0 (meaning there is no basement for that property),  Bsmt Qual can be "NA" because "NA" is used to fill the cell in the Bsmt Qual column if there is no basement.  Lot Frontage null values were filled by median values of Lot Frontage for the property's "Lot Shape" using the groupby method.  The Garage Yr Blt column was removed from the train dataframe because the Garage Yr Blt values were very hard to infer from contextual data of properties. 

The test set was cleaned using contextual information about properties just like the cleaning process for the train set, but rows were not removed.  Lot Frontage null values were filled by median values of Lot Frontage for the property's "Lot Shape" using the groupby method.  The Garage Yr Blt column was removed from the test dataframe as well because the Garage Yr Blt values were very hard to infer from contextual data of properties.  Also, the groupby method was used to fill a null Electrical value using the dataset's mode of the Electrical value for the row's corresponding Utilities value. The same thing was done to fill missing Garage Cond and Garage Finish values manually using the mode of Garage Cond when the Garage Type was detached.   

The cleaned train and test datasets' datatypes were checked, and the datasets were saved so they could be used in the next Jupyter Notebook.

#### Preprocessing: Initial Analysis and Adding Dummy Columns to Datasets and Building a Baseline Model
The cleaned train and test data were imported and were visually inspected and plotted/visualized via pairplots and heatmaps and boxplots.  Dummy dataframes were generated and appended to the test and train dataframes.  The dummy column with the largest number of 1's (rather than 0's) were removed from the test and train dataframes to remove collinearity and to increase visibility of unique dummy variables (which probably affect SalePrice more than the more common dummy variables which were removed).  The test and train dataframe then had different numbers of columns which meant that some dummy columns existed in the train set and not in the test set and vice versa, so dummy columns of 0's were added to each dataframe if they were missing.   

The baseline model was then built using all features in the train dataframe.  70% of the training data was used for training, and 30% of the training data was used for testing.  The SalePrice distribution was skewed, and data needed to be scaled, so the PowerTransform was used to transform data after the train-test-split was performed.  A linear regression model was built (Model 1), a linear regression model with LassoCV regularization was built (Model 2), a linear regression model with RidgeCV regularization was built (Model 3), and a linear regression model with ElasticNetCV regularization was built (Model 4).  For all of these models, the SalePrice predictions were interse-transformed to be able to return predictions to USD units (and to compare predicted SalePrice values to true SalePrice values). 

Hyperparameters of the LassoCV were chosen by allowing LassoCV to take in 200 alpha values with 5 cross-validation folds and to optimize the alpha value.  The optimal alpha value was used to generate the linear regression with LassoCV regularization and to then predict SalePrices.  The alpha hyperparameter for RidgeCV, alpha, was chosen by giving RidgeCV 100 values equally spaced from 0.01 to 10, and RidgeCV selected the optimal alpha value to use for the model (over the default number of cross-validation folds for RidgeCV).  For ElasticNetCV, 100 alpha values and 100 1L ratio values from 0.01 to 1 were passed through ElasticNetCV which optimised these hyperparameters to use for the model.

Each model's cross-validation score, train score, test score (before taking the inverse-transform of the predicted values), and test score (after taking the inverse-transform of the predicted values) were analyzed to determine how to move forward.  Note that at first, the predicted values were not reverse-transformed.  When the predicted values were reverse-transformed, the models performed much better in the Kaggle competition (especially because the variance dramatically decreased when the correct scale/unit were used).

#### Feature Engineering, Model Tuning, KaggleSubmissions, Metrics, and Visualizations
The baseline model did not perform well (especially when looking at the cross-validation scores).  The models were fit to too many features, so the correlation of features to SalePrice were inspected using a heatmap.  For Model 5, only features with correlation to SalePrice greater than 0.5 were used.  A linear regression model without regularization, a linear regression model with LassoCV regularization, a linear regression model with RidgeCV regularization, and a linear regression model with ElasticNetCV regularization were built, and the model with the best R squared scores (usually with a focus on the cross-validation scores) was chosen to submit to Kaggle.  The same methodology was used to build models, but this time without outliers.  The R squared scores surprisingly decreased, so the train data was used with outliers included after that.  

For Model 6, interaction features (including total bathrooms, years since built, years since remodeled, and BsmtFin SF 1+2) were tested, and if their magnitude of correlation to SalePrice increased compared to the features used to create the interactions, they were used for the model.  Only one of the tested interaction features, total bathrooms, had an increase in correlation to SalePrice, so it was used to build Model 6.  Model 6 improved in terms of R squared terms when compared to Model 5.

Model 7 included features with correlation to SalePrice greater than 0.3, but it did not improve as well as Model 6.  Therefore the features used in Model 6 were used moving on.  It would be interesting to find the optimal numer of features to use to build a better model though.

Model 8 included features from Model 6, and 75% of training data was used to build the model rather than the 70% of training data that was used to build previous models.  Model 8 did slightly better tha Model 6 when it came to cross validation score and other scores.  Model 8 became the final model due to the good R squared scores, and the MSE, RMSE, and MAE were all reasonable.  The predicted values vs. true values were plotted, and there appeared to be a pretty good fit between the true and predicted values (the scatter plot values were relatively close to the predicted values = true values line).  

It would have been interesting to optimize the number of features (and the actual features) to create a better model, to optimize the train-test-split ratios (ratio of data used for training vs. testing), to use a linear regression model to fill Lot Frontage values instead of using the groupby method during data cleaning (something that was attempted but not successfully completed), and to use other optimization techniques to make a better model.

### Sources:

City of Ames, https://www.cityofames.org/about-ames/interesting-facts-about-ames  

Dean De Cock, http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

https://www.google.com/search?biw=1280&bih=583&tbm=isch&sa=1&ei=pj2UXOzBPIO7jwS3_ZfgDQ&q=birds+eye+view+ames%2C+iowa+current&oq=birds+eye+view+ames%2C+iowa+current&gs_l=img.3...3318625.3321015..3321206...0.0..0.0.0.......13....1..gws-wiz-img.jSwJteC57P8#imgrc=Klagv1qes1xiKM: 
