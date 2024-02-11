# Preprocessing Plans
![](/images/df.png)
![](/images/dfdtypes.png)

A quick check of the data types present in our original dataset reveals that most of the data is not a numerical value. Fortunately, most of them can be converted into one with minimal work. We have outlined our thought processes below:



## Encoding

There are a couple features that are easy for us to convert due to the fact that the numerical values we need already exist within the entries, but are surrounded by characters. This causes pandas to treat them as objects.

* Visually, it seems that `Percent Recommended Class` and `Percent Recommended Class` are basically numbers, but since there's a % appended to them, pandas thinks they're strings/objects. We can fix those by just replacing each of these values with their numerical equivalent (i.e. 50% = 0.5) by stripping away the % using `Series.str.rstrip()` or `Series.str.extract()`.

* `Average Grade Expected` and `Average Grade Received` are also basically numbers, but since there's a letter grade prepended to them, pandas also thinks they're strings/objects. We can easily fix this by just stripping the letter grade and parentheses for all observations using the same methods listed above.

The remaining features don't have any obvious ways of converting the observations into numerical values, but we can figure out a way to do so by encoding the values that are there into numbers. Let's tackle course and quarter since they at least have some numbers in them.

* Observe that the entires under the `Quarter` feature are all in the format `XXYY` where `XX` refers to the quarter and `YY` refers to the year a particular course was offered. Since `XX` is very limited (Fall, Winter, Spring, SS1, SS2, Special), we can perform one hot encoding . Since we don't want to lose `YY` in the process, we can isolate it and make it its own feature called `Year`. This can be achieved by first separating `XX` and `YY` into their own dataframes `xdf` and `ydf` by using `Series.str.extract()`, then using the `OneHotEncoder` from `sklearn.preprocessing` to perform one hot encoding for `xdf` (call this `xdf'`). Then we can just concatenate `xdf` and `ydf` to our original dataframe after making sure they are all typecasted to a number.

* Next, we can observe that entries under the `Classes` feature are basically all the the format `Department + Course Number + Course Name`. Since we really only need the department name and course number to identify a class, we can safely ignore the course name. Thus, we can do the same thing that we did above and use `Series.str.extract()` to separate the department and the course numbers into their own data frames. However since some courses may have a suffix assigned to them (MATH 20A), we'd need to extract that too. The course numbers themselves can safely just be converted into integers, but in order to not lose the distinction between sequential classes (A,B,C,D), labs (l), remote classes (R), etc. we'd also want to create a new feature called `Suffix` and label encode the suffixes in alphabetical order (A = 0, B = 1, ...) as there can be many of them. Likewise since there's so many departments, we'd want to just use label encoding to assign a numerical value to each unique department, also based on alphabetical order. This can be achieved easily by just taking all the unique values in the dataframe for departments and suffixes and then assigning them a value and putting them in a hashmap, then calling `DataFrame.replace()` with the hashmap as an argument, in turn replacing all values with their corresponding encoded value.


The remaining columns `Instructor` and `Evaluation URL` can also be encoded. However, we feel that they aren't actually essential for the purposes of our project. This is because Evaluation URL just contains links to the CAPEs report that corresponds to a particular entry. While it has a bit more information that isn't in the dataset, the dataset already has all the key information pertaining to class and enrollment demographics, which is what we are interested in in the first place. Thus, we can drop this column from out dataset. Additionally, we feel that it is appropriate to exclude instructor names from our analysis for privacy and ethical reasons. This is because out goal is to develop a model that the school can use to understand and improve student success, enjoyment,and engagement and we wouldn't want a particular professor to have an influence on our model as that might leaf to favoritism or other workplace related issues. Since we already have a measure of how much the students like their professor in `Percentage Recommended Professor`, we can safely strip away the name and just have a measure of how much students enjoy their professor without having any names attached.

  
## Imputing

When using the .isna() function to view NA values in our dataset, we see that we have NA values in the Study Hours per Week, Average Grade Expected, and Average Grade Received features. How are we going to handle these NA values?

In our dataset, we have one NA value for `Study Hours Per Week`, so we will determine the course that has the NA value for Study Hours Per Week and then take the mean study hours for all other iterations of that course and replace the NA value with that number.

For the `Average Grade Expected` and `Average Grade Received` features, we have a significantly greater number of NA values 1486 and 17628 respectively. To handle these cases, we will create a map that has the department, course, and suffix of unique courses and then a column that includes the mean of the average grade expected and the mean of the average grade received for all iterations of that unique course. Then, we will iterate through the original dataframe, replacing any NA values in Average Grade Expected and Average Grade Received features with the mean values in our mapping for that course. This should be okay since students tend to perform relatively similarly across quarters, so our estimated value will be a fair approximator of how students actually might have performed during that particular iteration of the class taking into account any variability in performance caused by professors or external conditions (i.e. strikes).
