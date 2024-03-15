# **Introduction**
Our project leverages the power of artificial neural networks to investigate the relationship between various classroom metrics and their effects on college student success. Specifically, our project seeks to define a generalized model to predict how well students may perform in a college level course (as measured GPA) when provided data about the course and its professor.

As students ourselves, we feel that this project has a lot of value because the insights gained from it can directly impact our academic journeys and those of our peers. By understanding the factors that contribute to success in college courses, we can make more informed decisions about our education, such as selecting courses with professors known to facilitate student learning effectively. Additionally, the ability to predict student performance based on various metrics can provide early intervention opportunities for students who may be at risk of struggling academically, ultimately supporting retention and graduation rates.

Moreover, we think that it’s cool that we get to utilize a dataset that many of us and former students have contributed to throughout our time at UCSD (the CAPEs dataset) and build something out of it that can benefit students in the future.

The importance of this study comes from being able to provide a model to guide academic policy decisions which can enhance the quality of education, and foster an environment that supports and maximizes student success and learning enjoyment. As students ourselves, this project provided an opportunity to see how to utilize code and data related to academic performance to learn more about our own academic performance and uncover patterns in the data that can help us know what to look for to increase odds of student success. Having a good predictive model is important to highlight important patterns and behaviors that might not be clear to the human eye but are evident in the data. In other words, a good predictive model can be useful to develop improvements and predict future outcomes by uncovering patterns in the data.

# **Methods**
* ## Data Exploration:
    The dataset was explored using a combination of functions available through the pandas and seaborn libraries. Below is a summary of what functions were used and what we learned about our dataset from them
    * df.info() revealed that most features in our dataset contained objects (strings) instead of numeric values
    * pd.describe() revealed that approximately 27% of our dataset had missing values
    * Creating a correlation heatmap using df.corr() and sns.heatmap() revealed that:
        * `Average Grade Expected` is very strongly correlated (positively) with `Average Grade Received` 
        * `Study hours per week` is moderately negatively correlated with  `Average Grade Expected` and  `Average Grade Received`
        * `Percentage Recommended Class` and `Percentage Recommended Professor` is moderately positively correlated with `Average Grade Expected` and  `Average Grade Received`
        * Total enrolled in course was weakly negatively correlated with `Average Grade Received`
    * Plotting the distribution of the values using hist() revealed the data was largely skewed in one direction for most features. 

* ## Pre-Processing:
    The following techniques were used to process the CAPEs dataset:
    * ### Encoding:
      * Label encoding was performed on the columns `Percentage Recommended Class`, `Percentage Recommended Professor`, `Average Grade Expected` and `Average Grade Received` to convert them from strings into numeric values
      * The feature `Course`, containing details of a course’s department and course number, was processed to separate out the department, course number, and course suffix into their own columns. These individual features were then label encoded into numeric values and appended to our dataset under the names `Department`, `Course Number` and `Course Suffix` respectively.

    * ### Imputation:
      * The following strategy was used to impute missing values:
        > “For every observation with a missing value for feature X, create a value for X by taking all observations with the **same department, course number, and suffix** (the same class), taking the values reported for feature X, taking the average of them, and setting that as the imputed value for X.

      * For any remaining observations with missing values, the following strategy was used:
        > “For every observation with a missing value for feature X, create a value for X by looking at all observations within the same **department**, then taking the values reported for feature X, taking the average of them, and setting that as the imputed value for X. If this is not possible, take the average of all observations in the data set for feature X and replace the missing value with that"
 
      * The following features were impacted by imputation: `Study Hours per Week`, `Average Grade Expected`, and `Average Grade Received`

      * Imputing missing data saved **~27%** of the dataset from being dropped for having missing values 
  
    * ### Normalization:
      * Min-Max normalization was applied to all non-categorical data after the encoding and imputing steps
      * This impacted the following features: `Total Enrolled in Course`, `Percentage Recommended Class`, `Percentage Recommended Professor`, `Study Hours per Week`, `Average Grade Expected`, `Average Grade Received`, `Total CAPEs Given`

    * ### Dropped Columns:
      * The features `Instructor` and `Evaluation URL` were dropped from our dataset
      * The feature `Course` was dropped from the dataset after being processed and expanded into `Department`, `Course Number`, and `Course Suffix` features.

* ## Model 1:
  Model 1 consists of a DNN aimed at doing regression. It takes as input, values for the features `Total Enrolled in Course`, `Percentage Recommended Class`, `Percentage Recommended Professor`, `Study Hours per Week`, and `Average Grade Expected` and outputs a continuous value for `Average Grade Received`. It is a dense, sequential model configured with 5 hidden layers with a decreasing number of nodes all using the relu activation function. It was compiled using mean squared error as the loss function using the adam optimizer.

  The code for the model can be found below:
  ```
  model_1 = Sequential([
   Dense(units = 32, activation='relu', input_dim = X_train.shape[1]),
   Dense(units = 16, activation='relu'),
   Dense(units = 8, activation='relu'),
   Dense(units = 4, activation='relu'), 
   Dense(units = 2, activation='relu'), 
   Dense(units = 1) 
  ])
  
  model_1.compile(optimizer='adam', loss='mse')
  ```
  The model was trained on 30 epochs using a batch size of 10 with early stopping and model checkpointing enabled to save the weights that resulted in the lowest evaluation loss.

  The model was evaluated for performance and signs of overfitting by plotting training vs evaluation loss and by comparing the model’s training error (mse) with that of its testing error. 

* ## Model 2:
  Model 2 consists of a DNN aimed at doing regression. Like Model 1, it is a dense, sequential DNN outputting a continuous value for `Average Grade Received` but takes additional features on top of what model 1 already takes as input: namely `Department`, `Course Number`, and `Course Suffix`. Model 2 is also different in that hyperparameter tuning has been performed on its activation functions and optimizer with the objective of minimizing loss. The optimal results were then extracted and used to construct model 2.
  
  The code for the model (post hyper-parameter tuning) can be found below:
  ```
  model_2 = Sequential([
   Dense(units = 32, activation='tanh', input_dim = X_train.shape[1]),
   Dense(units = 8, activation='tanh'), 
   Dense(units = 4, activation='tanh'),
   Dense(units = 2, activation='relu'),
   Dense(units = 1) 
  ])

  model_2.compile(optimizer='adam', loss='mse')
  ```

  The model was trained on 30 epochs using a batch size of 10 with early stopping to save the weights that resulted in the lowest training loss.

  Like model 1, model 2 was evaluated for performance and signs of overfitting by plotting training vs evaluation loss and by comparing the model’s training error (mse) with that of its testing error. K-fold cross validation was also performed for the same purpose.

* ## Model 3:

  Model 3 consists of a DNN aimed at doing classification. Unlike the previous models, it aims to predict the letter grade of a class given the inputs `Total Enrolled in Course`, `Percentage Recommended Class`, `Percentage Recommended Professor`, `Study Hours per Week`, `Average Grade Expected` instead of a continuous value. To do this, the dataset was modified to convert the GPA values in `Average Grade Expected` into letter grades, which were then subsequently one hot encoded. Thus, model 3 serves as a DNN that predicts multi class outputs.

  Model 3 is configured to be a dense, sequential model with 5 hidden layers all with the relu activation function and a decreasing number of nodes. To perform multi class classification, the output layer uses the softmax function and the model is compiled with the categorical cross entropy loss function with rmsprop as the optimizer. 

  The code for the model can be found below:
  ```
  model_3 = Sequential([
   Dense(units = 512, activation='relu', input_dim = X_train.shape[1]),
   Dense(units = 256, activation='relu'),
   Dense(units = 128, activation='relu'),
   Dense(units = 64, activation='relu'), 
   Dense(units = 32, activation='relu'), 
   Dense(units = y_train.shape[1], activation = "softmax")
  ])

  model_3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
  ```

  As with the previous model, model 3 was trained on 30 epochs using a batch size of 10 with early stopping to save the weights that resulted in the lowest training loss.

  The model was evaluated for performance and signs of overfitting by plotting training vs evaluation loss and by comparing the model’s accuracy, recall, and precision for various classes using sklearn’s classification report for both testing and training data.

* ## Alternative Models
  As an exercise, we've prepared several models using various other concepts covered in class. While they are not our main model, these models serve as a base of comparison for our DNNs so we have another way to compare their performance. So as to increase the readibility of the README however, detailed descriptions, code, and graphs related to these models will be excluded, though we will provide a link to them in the notebook.

  A brief summary of what they are has been provided below for completeness:
    * Regression:
      * Version 1:
        * sklearn's LinearRegression() model was used to create a linear regression and was evaluated using K-fold cross validation of varying degrees.
      * Version 2:
        * Hyperparameter tuning was performed to evaluate the LinearRegression() model's performance when given different inputs. The best one was then evaluated using K-fold cross validation of varying degrees.
      * Version 3:
        * Used sklearn's PolynomialFeatures() to construct regression models of degree ranging from 1-5
    * KNN:
      * Version 1:
        * Used sklearn's KNeighborsClassifier() to construct a KNN model with K = 50
      * Version 2:
        * Used sklearn's KNeighborsClassifier() to construct a KNN model with K = 200
      * Version 3:
        * Used sklearn's KNeighborsClassifier() to construct a KNN model with K = 500

# Results
* ## Data Exploration:
  * df.info() revealed that most features in our dataset contained objects (strings) instead of numeric values
    * ![](images/results_info.png)
  * pd.describe() revealed that approximately 27% of our dataset had missing values
    * ![](images/results_missing.png)
  * Creating a correlation heatmap using df.corr() and sns.heatmap() revealed the following correlation coefficients: 
    * ![](images/results_corr.png)
  * Viewing the distribution of the values in each feature revealed the following skews
    * ![](images/results_dist.png)
  
* ## Pre-Processing:
    * ### Encoding:
      * Prior to encoding our dataset, the majority of the features contained objects as their value.
      * After encoding the features mentioned in the methods section, all of the features in the dataset were converted to either integers or floats
      * A before and after view of the dataset can be found below
        * Before:
        * ![](images/results_encoding_before.png)
        * After:
        * ![](images/results_encoding_after.png)
    * ### Imputation:
      * Applying the technique outlined in the methods section reduced the number of rows with missing data from 19,115 all the way to 0.
    * ### Normalization:
      * Min Max normalization was applied to non-categorical columns. The resulting dataset can be seen below:
        * ![](images/results_normalization.png)
    * ### Dropped Columns:
      * The features `Instructor` and `Evaluation URL` were dropped from our dataset
      * The feature `Course` was dropped from the dataset after being processed and expanded into `Department`, `Course Number`, and `Course Suffix` features.
      * This was done while encoding the data. Thus, the previews of the dataset used to show the state of the dataset after the applying the different techniques exclude these values.
* ## Model 1:
    * Training vs Validation Loss Graph:
      * ![](images/results_model1.png)
    * Training vs Test MSE:
      * ```
        MSE Train: 0.012811824793743347
        MSE Test: 0.012531321231611235
        ```
* ## Model 2:
    * Training vs Validation Loss Graph:
      * ![](images/results_model2.png)
    * Training vs Test MSE:
      * ```
        MSE Train: 0.006345696507123405
        MSE Test: 0.006237075743043597
        ```
    * Losses per Fold for K-Fold Cross Validation 
      * ```
        array([0.00620253, 0.00730672, 0.00649318, 0.00584978, 0.00591005])
        ```
* ## Model 3:
    * Training vs Validation Loss Graph:
    * Training vs Test Classification Report:
* ## Alternative Models
    * Graphs and figures for our alternative models can be found [here]()

# Discussion
* ## Data Exploration:
* As our goal is to build a DNN that predicts student grades, we felt that it was first important to explore our dataset to find any correlations between other features and our target so that we could determine what inputs to include in our model. Thus, we made use of the corr() method in pandas to compute the correlation coefficient between all the features in our dataset and plotted them using heatmap() from seaborn with a diverging color scale. This made it easy to visually spot which features were more strongly correlated with “Average GPA Received” and whether the association was positive or negative. After doing so, we identified that the features “Average Grade Expected”, “Study hours per week”, “Percentage Recommended Class”, “Percentage Recommended Professor” had strong or medium levels of correlation with our target, which gave us our inputs.
> * Note: we also decided to add the feature “Total Enrolled In Class” as an input even though it had a weak negative correlation (-0.22) with our target. This came from discussion about whether or not students generally do better in smaller courses or ones with a larger amount of students. Thus, we felt that it was a good idea to also include it as a possible input for our model. In retrospect, we probably should have only selected features with a medium or strong correlation to our target feature as that would’ve likely made it easier for our model to achieve better predictive performance (by eliminating noise caused by features that aren’t that “useful”).

* ## Pre-Processing:
    * ### Encoding:
      * Encoding was performed because the majority of the features in our dataset contained strings as values. Thus, in order to use them as inputs for our models, we needed to turn them into numbers first. Fortunately, many of them were easy to convert into a numeric value because the number we wanted was already contained within the string. This meant that we were able to write some RegEx to extract the numeric values we needed from the problematic columns. We then overwrote those columns with their extracted numeric values. This was applied to the columns “Percentage Recommended Class”, "Percentage Recommended Professor”, “Average Grade Expected” and “Average Grade Received”.

      While these were the only features we really needed to encode, we figured that we might as well encode some extra features in the dataset in case we needed to use them later. This led us down a path to encode the “quarter” and “course” feature, which was a bit more complicated. 
        
      To encode “Quarter”, we noticed that the entries under this feature were all in the format XXYY where XX referred to the quarter and YY referred to the year a particular course was offered. This meant that we could again write some RegEx to separate out the quarter and year into their own features “Quarter” and “Year”. As “Year” was already a number, we simply added 2000 to it to make it more interpretable.  To handle ”Quarter” we just performed label encoding on it as quarters tend to have an order associated with them (fall -> winter -> spring…).

      To encode “Course” we observed that entries for this feature were basically all in the format Department + Course Number + Course Name. Since we really only needed the department name and course number to identify a class, we realized that we could just ignore the course name. This meant that we could encode values in “Course” by separating out the department and the course numbers into their own columns (department and course numbers respectively). However, since some courses had a suffix assigned to them (MATH 20A), we also had to extract that too and create a new column, suffix. As with before, this was achieved with RegEx. From there, the course numbers themselves could safely just be converted into integers, but we had to encode the department and suffixes as they were entirely strings. Since there were quite a few departments and suffixes, we just label encoded them by assigning a numeric value to each based on their lexicographic ordering instead of doing one hot encoding to reduce the number of features in our dataset.

    * ### Imputation:
      * The choice to try to impute missing data was driven by the discovery that around ~27% of our dataset contained missing values. While we considered dropping them since our dataset was sufficiently large enough, we decided to try to impute as many entries as possible before resorting to dropping.
      * Once we learned that the bulk of the missing values were for the columns ‘Average Grade Expected and “Average Grade Received” we reasoned that we could fill those values by looking at other instances of the same class (as identified by having the same department number, course number, and course suffix) and replace the missing value with the average of the values from said classes. This technique managed to reduce the number of observations with missing values to <10% of our dataset. While we figured we could just stop there, we decided to look a bit further into it in between milestones and discovered that some classes only occurred for one instance. This meant that we couldn’t compute a value to replace the missing one. Fortunately, we realized we could just utilize the same technique as before, but with a slight modification. While less ideal, we could just look at other classes in the entire department and generate any replacement value by averaging the values observed in said classes. Failing that (i.e. if the department only had the class with the missing value) the entire dataset,we could still compute a value by taking the average of all the values for the feature with a missing value. This basically served as a catch all filter that imputed all of the data with missing values in our dataset! Thus, we were able to avoid the need to drop any entries from our dataset.

    * ### Normalization:
      * We chose to min-max normalize non-label values in our dataset as it was mentioned in class that normalizing data was necessary if we ever needed to perform stochastic gradient descent optimization since it is sensitive to scale.

      * The decision to choose min-max normalization in particular was made because some of our features already appeared to be min max normalized once we converted them into numeric values (“Percentage Recommended Class” and "Percentage Recommended Professor all had values between 0 and 1), thus we figured we might as well choose to min max normalize the rest of our data.
  
    * ### Dropped Columns:
      * The features “Instructor and Evaluation URL” were dropped from our dataset. This was because “Evaluation URL” just contained links to the CAPEs report that corresponded to a particular entry. While the report had a bit more information that wasn’t in the dataset, what we had already contained all the key information pertaining to class and enrollment demographics, which is what we were interested in in the first place. 
      * Additionally, we felt that it was appropriate to exclude instructor names from our analysis for privacy and ethical reasons. This is because our goal is to develop a model that the school and other students can use to understand and improve student success (as measured by grades) and we wouldn't want a particular professor to have an influence on our model as that might lead to favoritism or other workplace related issues. Since we already have a measure of how much the students like their professor in Percentage Recommended Professor, we figured we could safely strip away the name and just have a measure of how much students enjoy their professor without having any names attached.

* ## Model 1:
* ## Model 2:
* ## Model 3:
* ## Alternative Models

# Conclusion


# Collaboration