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

* Next, we can observe that entries under the `Classes` feature are basically all the the format `Department + Course Number + Course Name`. Since we really only need the department name and course number to identify a class, we can safely ignore the course name. Thus, we can do the same thing that we did above and use `Series.str.extract()` to separate the department and the course numbers into their own data frame. The course numbers can safely just be converted into integers. To not lose the distinction between sequential classes (A,B,C,D), labs (l), and remote classes (R) however, we'd want to one hot encode them as there aren't many class number suffixes. That being said, since there's so many departments, we'd want to just use normal label encoding to assign a numerical value to each unique department, perhaps in based on alphabetical order. 
  
## Imputing
