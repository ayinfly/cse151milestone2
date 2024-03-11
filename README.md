# Table of Contents



# Milestone 2 Notes:
## Preprocessing
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


## Normalization
We will be normalizing our input features using the min-max normalization technique in order to improve the performance of the regression ANN we will be producing to predict student success (i.e. average grade received), enjoyment (i.e. % recommend the class), and engagement (i.e. study hours per week).





# Milestone 3 Notes:

### Note: more detailed responses and graphs are in the notebook

## [Finish major preprocessing](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=ql2uSH6ICs2l)

All of the steps outlined in the preprocessing plans section above has been completed. We did run into some issues in imputing, namely in instances where when we tried to compute the mean of all iterations of a particular class with NaN values, we would get another NaN value because ALL other iterations of that class also had NaN values for the feature we are trying to take the mean of. For observations (classes) with these properties, we have addressed them by taking the mean of all classes in that department. Despite doing so, there were still a few classes with NaN values for some feature columns. So we remedied that by just assigning it the value equal to the mean of all observation values in the dataset for the column with the issue. This has reduced the number of NaN values in our dataset to 0.

## [Train your first model](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=rfSryuY1OVNC&line=1&uniqifier=1)
Our first model is an DNN with 5 hidden layers all using the Relu activation function and with a decreasing number of nodes in each layer. Since we are doing regression where we need to predict a continuous value, the output layer has no activation function. We will be using the adam optimizer and mse as our loss function. Early stopping and checkpointing has also been set up for the model's fit function.

## [Evaluate your model compare training vs test error](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=vycnE1NkOfQx&line=1&uniqifier=1)
Our model's initial performance was quite good. Our model managed a MSE loss value of 0.006070451192523905 on the training data and a value of 0.0059246622968575175 on the test. Thus, it looks like our model performs pretty similarly when presented with unseen test data as it does with the data it was trained on. In fact, the results seem to indicate that the model performs slightly better with the provided test data than the training data as the test MSE is slightly lower than the training MSE. We belive this is probably just due to luck (the seed that we used to split the data) and that if we performed k cross fold validation we may see slightly different results. Ultimately, though the loss is not perfect (MSE is not 0) for either dataset, the loss values that we do get are rather low, which suggests that our model would do a good job at preducting the average GPA received for a class given the values for "Total Enrolled in Course", "Percentage Recommended Professor", "Study Hours per Week", and "Average Grade Expected". These are quite promising results for our first model!

## [Where does your model fit in the fitting graph?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=vycnE1NkOfQx&line=1&uniqifier=1)
![](/images/model_1_graph.png)
Plotting the training vs validation loss reveals that our model mostly lies in the "Good fit" section of the Underfitting/Overfitting graph. In our plot, we can see that as the number of epochs increases, the training loss continues to get lower while the validation loss largely follows suit. While there is some divergence (as evident by the varying peaks and valleys in the validation loss), they do not deviate too far from the training loss and show a general trend downwards that mirrors it as well. Likewise, since the loss values are so low and are appearing to stabilize at a low value for both the training and validation data, this tells us that we are not underfitting, and since the validation loss does not start to consistently increase while the training loss decreases as the number of epochs increases, it appears that overfitting is not occurring to a significant degree as well, which means that our model sits somewhere in the good fit zone.

## [What are the next 2 models you are thinking of and why?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=wkSDc_koOj9p&line=1&uniqifier=1)

Per our milestone 1 submission, we aim to create 2 additional models from our data set, both of which are also regression problems. The first alternative model is one that attempts to predict how much students will enjoy a class (as determined by the % who recommended the class) given some of the other available features (i.e. "Total Enrolled in Course", "Percentage Recommended Professor", "Study Hours per Week", "Average Grade Expected", etc.), as this is one way to measure student enjoyment in a class. The other model we are planning on developing is one to predict student enagagement, particularly by estimating how many hours they would likely need to spend per week studying. Like the other model, we will also use some of the other features avaiable in our dataset as inputs to our model. This time, we will also use class department and upper/lower div classification as we know that class difficulty and study hours can differ greatly across departments and classes.

## [Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=i5FkYAmuOss6&line=3&uniqifier=1)

Overall, it seems that our first model is rather successful at predicting average GPAs received as evident by the low loss of its outputs in both training, test, and validation data. However, we feel that there is more work that we can do. Here is a list of some of the plans that we have to improve the performance of our model:

- Hyperparameter tune. Mess around with different numbers of nodes in each layer, activation functions, loss functions, optimizers, and learning rates. Since we have some time before the next milestone is due, we can definitely have a member of our group tune our existing model to see if changing anything can help us reduce our loss

- Add extra features to the dataset. One of our teammates realized that the evaluation urls present in the orignal dataset takes you to the CAPEs review for the observation it's associated with and contains a bunch of other data that we feel might give our model more relevant information that it can use to make better predictions. They are currently working on a webscraping script to extract such data so that we can add it to our dataset.

- Run more epochs and increase the early stopping threshold. The trend in the training data's loss values as the number of epochs increases suggests that the model can be trained for more epochs to reduce the training loss. More work would need to be done to verify that overfitting does not happen or appear to get worse as a result of this however.

- Cross fold validation. We can perform K fold validation to test our model's degree of overfitting. If we notice that the MSE for different folds varies greatly then we know that our model may be overfitting, which may reduce its performance when presented with unseen data.

- Incorporating the discretized labels as input features. Currently our model is class agnostic, meaning that it does not consider department and class number (upper/lower div classification) at all. We plan on incorporating that as we feel that it is important as GPAs do differ from department to department.

- Changing the normalization type. Standardizing the input data may result in better performance. We will experiment with this for the next milestone.

# Milestone 4 Notes:

## [Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=IbOaMV6pfldE)

Looking at the performance of our first model, it would seem that our initial configuration of the data set and DNN yielded promising results. Our reported mse loss (for both training and validation) was very low (<= 0.0128), so we can probably just get away with leaving the configuration as is, but we think we can do better. To achieve this, we are planning on performing Hyperparameter tuning and K-fold cross validation to improve our existing model and check that it was not overfit to our traning data or that our seed for the test data wasn't just very lucky. Moreover we'll be incorporating other features that we left out of the training/test dataset used for model 1 that we think may help to reduce the loss. Namely, we'll try to incorporate the department and class number (and suffixes) into our dataset as we know intuitively that different departments have varying difficulties and higher number classes (upper divs) tend to be harder than lower divs. Likewise, certain courses in a series may be harder than others, so incorporating the course suffix may help our model's peformance.

Thus, we will be:

- changing our data to include more features to increase differentiability
- keeping our loss function the same (mse) as it seemed to produce good results in model 1
- keeping the labels the same (technically there aren't any labels since our output values are continuous values and we won't be changing it in any way)


## [Train your second model](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=Zy-bSuGifx_j)

To train our second model, we first performed hyperparameter tuning on our model from the previous milestone. Given time constraints, we opted to only tune the activation functions for each layer and optimizers. This resulted in 87 unique permutations to assess. Running the hyperparamter tuner yielded that the optimal combination was (from the first to last hidden layer) tanh, tanh, tanh, tanh, relu with the adam optimizer. These findings were then used to construct model 2 of our GPA predictor DNN.

## [Evaluate your model compare training vs test error](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=qVxYXsjsf0UE)

Our new model's training and test error were surprisingly similar, but much better than what we got for model 1. In fact, evaluating the training and test MSE for this model reveals the following: 
```
MSE Train: 0.008263694997974086
MSE Test: 0.008107799167541802
```

Furthermore, the results of doing K-fold cross validation with 5 folds gives us these scores:
```
[0.00672934, 0.0063089 , 0.00598459, 0.00601308, 0.00585237]
```

Taking all of these into consideration, our 2nd model exhibits about a 1.5-2x reduction in the MSE compared to our model for milestone 3! Additionally, this consistency in K-fold cross validation suggests that our model is not overfitting and is working well with unseen data, which is exactly what we want!

## [Where does your model fit in the fitting graph, how does it compare to your first model?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=BcBOnZd9f8VN)

![](/images/model_2_graph.png)

While there is some divergence (as evident by the varying peaks and valleys in the validation loss), they do not deviate too far from the training loss and show a general trend downwards that mirrors it as well. Likewise, since the loss values are so low and are appearing to stabilize at a low value for both the training and validation data, this tells us that we are not underfitting. Morever the validation loss does not show a consistent increase while the training loss decreases as the number of epochs increases. Overfitting may be starting to occur as evident by the small increase in the val_loss in the final epoch, but it could also just be a one off event (there are several small spikes upwards in the val_loss). Thus, we'd argue that overfitting is probably not occurring to a degree significant enough to impair the generalizability of our model, which means that our model sits somewhere in the good fit zone, much like what we observed in our model 1!

## [Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=ZN57u13ogH5C&line=3&uniqifier=1)

We performed hyperparameter tuning and K-fold validation but no feature expansion (technically we did increase the number of features by adding features that were already in our dataset, but didn't create any new ones by raising values by a power). Hyperparameter tuning was done to tune the activation function for each layer and the optimizer and K-fold validation was performed with 5 splits. The results of doing so reduced our training mse loss from ~0.0128 to 0.008, approximately a 1.6x reduction! Not only that, but the results of the K-fold cross validation reveals that the model's performance on various test datasets hovers around the same value. This is promising as it means our model and its weights generalizes well to unseen data.

## [What is the plan for the next model you are thinking of and why?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=Vq8Lzd6KgLGA)

We're going to experiment with creating a classification DNN instead of a regression one. Technically, we could proceed with another regression DNN and perform more hyperparameter tuning, but the loss is so low already it might not be worth it (not to mention the exponential amount of time it would take to expand the tuning parameters). Moreover, we think the reason why the loss is really small is because the range of possible output values is quite small to begin with (something between 0-4). Thus, whatever our model predicts is likely to not be that far off from the true value from a relative sense, which may be throwing off our interpretation of the model's performance. Besides, when students check capes to evaluate classes and professors, they don't really care about the specific GPA received, but rather the letter grade. So for our 3rd model, we're planning on making some modifications to our data set (label encoding) and changing the DNN's architecture (i.e. the loss function) in order to perform classification. The target will still be GPA, but we will first need to label encode it into discrete values in order to make it something that we can predict (since letter grades are ordinal). Thus, the goal with the third model is to see if we can still produce a model that accurately predicts the grade letter received, rather than the specific GPA. Like model 2, we will determine what the optimal activation functions and optimizers are for model 3 using hyperparameter tuning and evaluate overfitting using K-fold validation.

## [Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?](https://colab.research.google.com/drive/1fSYLGAT1rz91a4LCf_CJ20AT7SilrLFe?authuser=1#scrollTo=SDel_O1FgPaQ)

Overall, Model 2 appears to be an improvement compared Model 1 since we were able to reduce the training MSE loss by 1.5-2x. This was accomplished by

- extending our input data to also take into account a course's department, course number, and sourse suffix for extra differenitability.

- utilizing hyperparameter tuning to determine the optimal selection of activation functions and the error optimizer, which were then used in the construction of this milestone's DNN.

- evaluating overfitting by using K-fold validation which revealed that our model was performing similarly to unseen testing data from fold to fold, a good sign that overfitting was not significantly occurring. (this was not done to the model in milestone 3)

Together, this seems to indicate that our model is an overall improvement to model 1, becoming more "accurate" (reduced loss), while simultantously still remaining generalizable, which is exactly what we wanted to happen! We imagine that hyperparameter tuning did the bulk of the work for us by brute force testing different combinations of the activation functions and optimizers so that we could figure out which ones were best and in what order.

We'd imagine that we could further optimize the performance of this model by performing more hyperparameter tuning. Due to time constraints, we could only let the tuner run for a little over an hour, but we'd imagine that if we had more time (and if google decides to not time us out), we could tune more parameters like the number of layers, the number of nodes in each layer, more activation/loss functions, and the learning rate to further reduce the loss. That being said, since the loss is already so low, we'd imagine that further optimizations would follow the law of diminishing returns. Thus, it would be better to consider a different kind of model (i.e. classification) to predict grades instead and see how it stacks up to this one.