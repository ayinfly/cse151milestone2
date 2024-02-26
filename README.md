# Preprocessing
![](/images/df.png)
![](/images/dfdtypes.png)

A quick check of the data types present in our original dataset reveals that most of the data is not a numerical value. Fortunately, most of them can be converted into one with minimal work. We have outlined our thought processes below:



## Encoding

There are a couple features that are easy for us to convert due to the fact that the numerical values we need already exist within the entries, but are surrounded by characters. This causes pandas to treat them as objects.

* Visually, it seems that `Percent Recommended Class` and `Percent Recommended Class` are basically numbers, but since there's a % appended to them, pandas thinks they're strings/objects. We can fix those by just replacing each of these values with their numerical equivalent (i.e. 50% = 0.5) by stripping away the % using `Series.str.rstrip()` or `Series.str.extract()`.

* `Average Grade Expected` and `Average Grade Received` are also basically numbers, but since there's a letter grade prepended to them, pandas also thinks they're strings/objects. We can easily fix this by just stripping the letter grade and parentheses for all observations using the same methods listed above.

The remaining features don't have any obvious ways of converting the observations into numerical values, but we can figure out a way to do so by encoding the values that are there into numbers. Let's tackle course and quarter since they at least have some numbers in them.

* Observe that the entires under the `Quarter` feature are all in the format `XXYY` where `XX` refers to the quarter and `YY` refers to the year a particular course was offered. Since `XX` is very limited (Fall, Winter, Spring, SS1, SS2, Special), we can perform one hot encoding . Since we don't want to lose `YY` in the process, we can isolate it and make it its own feature called `Year`. This can be achieved by first separating `XX` and `YY` into their own dataframes `xdf` and `ydf` by using `Series.str.extract()`, then using the `OneHotEncoder` from `sklearn.preprocessing` to perform one hot encoding for `xdf` (call this `xdf'`). Then we can just concatenate `xdf` and `ydf` to our original dataframe after making sure they are all typecasted to a number.

* Next, we can observe that entries under the `Classes` feature are basically all the the format `Department + Course Number + Course Name`. Since we really only need the department name and course number to identify a class, we can safely ignore the course name. Thus, we can do the same thing that we did above and use `Series.str.extract()` to separate the department and the course numbers into their own data frames. However since some courses may have a suffix assigned to them (MATH 20A), we'd need to extract that too. The course numbers themselves can safely just be converted into integers, but in order to not lose the distinction between sequential classes (A,B,C,D), labs (l), remote classes (R), etc. we'd also want to create a new feature called `Suffix` and label encode the suffixes in alphabetical order (A = 0, B = 1, ...) as there can be many of them. Likewise since there's so many departments, we'd want to just use label encoding to assign a numerical value to each unique department, also based on alphabetical order. This can be achieved easily by just taking all the unique values in the dataframe for departments and suffixes and then assigning them a value and putting them in a hashmap, then calling `DataFrame.replace()` with the hashmap as an argument, in turn replacing all values with their corresponding encoded value. Together, these changes allow us to to encode the identifiable information for each class such that they can be used as inputs into our model.


The remaining columns `Instructor` and `Evaluation URL` can also be encoded. However, we feel that they aren't actually essential for the purposes of our project. This is because Evaluation URL just contains links to the CAPEs report that corresponds to a particular entry. While it has a bit more information that isn't in the dataset, the dataset already has all the key information pertaining to class and enrollment demographics, which is what we are interested in in the first place. Thus, we can drop this column from out dataset. Additionally, we feel that it is appropriate to exclude instructor names from our analysis for privacy and ethical reasons. This is because out goal is to develop a model that the school can use to understand and improve student success, enjoyment,and engagement and we wouldn't want a particular professor to have an influence on our model as that might leaf to favoritism or other workplace related issues. Since we already have a measure of how much the students like their professor in `Percentage Recommended Professor`, we can safely strip away the name and just have a measure of how much students enjoy their professor without having any names attached.

  
## Imputing

When using the .isna() function to view NA values in our dataset, we see that we have NA values in the Study Hours per Week, Average Grade Expected, and Average Grade Received features. How are we going to handle these NA values?

In our dataset, we have one NA value for `Study Hours Per Week`, so we will determine the course that has the NA value for Study Hours Per Week and then take the mean study hours for all other iterations of that course and replace the NA value with that number.

For the `Average Grade Expected` and `Average Grade Received` features, we have a significantly greater number of NA values 1486 and 17628 respectively. To handle these cases, we will create a map that has the department, course, and suffix of unique courses and then a column that includes the mean of the average grade expected and the mean of the average grade received for all iterations of that unique course. Then, we will iterate through the original dataframe, replacing any NA values in Average Grade Expected and Average Grade Received features with the mean values in our mapping for that course. This should be okay since students tend to perform relatively similarly across quarters, so our estimated value will be a fair approximator of how students actually might have performed during that particular iteration of the class taking into account any variability in performance caused by professors or external conditions (i.e. strikes).


# Normalization
We will be normalizing our input features using the min-max normalization technique in order to improve the performance of the regression ANN we will be producing to predict student success (i.e. average grade received), enjoyment (i.e. % recommend the class), and engagement (i.e. study hours per week).





# Milestone 3 Notes:

### Note: more detailed responses and graphs are in the notebook

## [Finish major preprocessing](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=ql2uSH6ICs2l)

    All of the steps outlined in the preprocessing plans section above has been completed. We did run into some issues in imputing, namely in instances where when we tried to compute the mean of all iterations of a particular class with NaN values, we would get another NaN value because ALL other iterations of that class also had NaN values for the feature we are trying to take the mean of. For observations (classes) with these properties, we have addressed them by taking the mean of all classes in that department. Despite doing so, there were still a few classes with NaN values for some feature columns. So we remedied that by just assigning it the value equal to the mean of all observation values in the dataset for the column with the issue. This has reduced the number of NaN values in our dataset to 0.

## [Train your first model](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=rfSryuY1OVNC&line=1&uniqifier=1)
    Our first model is an DNN with 7 hidden layers all using the Relu activation function and with a decreasing number of nodes in each layer. Since we are doing regression where we need to predict a continuous value, the output layer has no activation function. We will be using the adam optimizer and mse as our loss function. Early stopping and checkpointing has also been set up for the model's fit function.

## [Evaluate your model compare training vs test error](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=vycnE1NkOfQx&line=1&uniqifier=1)
    Our model's initial performance was quite good. Our model managed a MSE loss value of 0.006070451192523905 on the training data and a value of 0.0059246622968575175 on the test. Thus, it looks like our model performs pretty similarly when presented with unseen test data as it does with the data it was trained on. In fact, the results seem to indicate that the model performs slightly better with the provided test data than the training data as the test MSE is slightly lower than the training MSE. We belive this is probably just due to luck (the seed that we used to split the data) and that if we performed k cross fold validation we may see slightly different results. Ultimately, though the loss is not perfect (MSE is not 0) for either dataset, the loss values that we do get are rather low, which suggests that our model would do a good job at preducting the average GPA received for a class given the values for "Total Enrolled in Course", "Percentage Recommended Professor", "Study Hours per Week", and "Average Grade Expected". These are quite promising results for our first model!

## [Where does your model fit in the fitting graph?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=vycnE1NkOfQx&line=1&uniqifier=1)
    ![](/images/model_1_graph.png)
    Plotting the training vs validation loss reveals that our model mostly lies in the "Good fit" section of the Underfitting/Overfitting graph. In our plot, we can see that as the number of epochs increases, the training loss continues to get lower while the validation loss largely follows suit. While there is some divergence (as evident by the varying peaks and valleys in the validation loss), they do not deviate too far from the training loss and show a general trend downwards that mirrors it as well. Likewise, since the loss values are so low and are appearing to stabilize at a low value for both the training and validation data, this tells us that we are not underfitting, and since the validation loss does not start to consistently increase while the training loss decreases as the number of epochs increases, it appears that overfitting is not occurring to a significant degree as well, which means that our model sits somewhere in the good fit zone.

## [What are the next 2 models you are thinking of and why?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=wkSDc_koOj9p&line=1&uniqifier=1)

    Per our milestone 1 submission, we aim to create 2 additional models from our data set, both of which are also regression problems. The first alternative model is one that attempts to predict how much students will enjoy a class (as determined by the % who recommended the class) given some of the other available features (i.e. "Total Enrolled in Course", "Percentage Recommended Professor", "Study Hours per Week", "Average Grade Expected", etc.), as this is one way to measure student enjoyment in a class. The other model we are planning on developing is one to predict student enagagement, particularly by estimating how many hours they would likely need to spend per week studying. Like the other model, we will also use some of the other features avaiable in our dataset as inputs to our model. This time, we will also use class department and upper/lower div classification as we know that class difficulty and study hours can differ greatly across departments and classes.

## [Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=i5FkYAmuOss6&line=3&uniqifier=1)

Overall, it seems that our first model is rather successful at predicting average GPAs received as evident by the low loss of its outputs in both trianing, test, and validation data. However, we feel that there is more work that we can do. Here is a list of some of the plans that we have to improve the performance of our model:

- Hyperparameter tune. Mess around with different numbers of nodes in each layer, activation functions, loss functions, optimizers, and learning rates. Since we have some time before the next milestone is due, we can definitely have a member of our group tune our existing model to see if changing anything can help us reduce our loss

- Add extra features to the dataset. One of our teammates realized that the evaluation urls present in the orignal dataset takes you to the CAPEs review for the observation it's associated with and contains a bunch of other data that we feel might give our model more relevant information that it can use to make better predictions. They are currently working on a webscraping script to extract such data so that we can add it to our dataset.

- Run more epochs and increase the early stopping threshold. The trend in the training data's loss values as the number of epochs increases suggests that the model can be trained for more epochs to reduce the training loss. More work would need to be done to verify that overfitting does not happen or appear to get worse as a result of this however.

- Cross fold validation. We can perform K fold validation to test our model's degree of overfitting. If we notice that the MSE for different folds varies greatly then we know that our model may be overfitting, which may reduce its performance when presented with unseen data.

- Incorporating the discretized labels as input features. Currently our model is class agnostic, meaning that it does not consider department and class number (upper/lower div classification) at all. We plan on incorporating that as we feel that it is important as GPAs do differ from department to department.

- Changing the normalization type. Standardizing the input data may result in better performance. We will experiment with this for the next milestone.


