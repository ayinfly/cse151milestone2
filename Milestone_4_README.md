
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