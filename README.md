# Machine-Learning-Projects

A Repo for my AI, Data Science and Machine Learning Projects

## Content
1. California Housing Prices Predictor
2. MNIST Digits Classifier- (Rite of Passage)
3. Titanic Survival Predictor
4. Spam Message Classifier 
5. Description of some Machine Learning Algorithms

## Project 1: California Housing Prices Predictor

[Link to Dataset](https://raw.githubusercontent.com/ageron/handson-ml/master/)

[Link to Notebook](california-housing-prices-prediction\ch2_housing_prices_exercise.ipynb)

This data has metrics such as the population, median income, median housing price, and so on for each block group in California. 
Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). We will just call them “districts” for short. This model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

### Steps
1. Look at the Big Picture
2. Get the Data
3. Discover and visualize the data to gain insights.
4. Prepare the Machine Learning Algorithms
5. Select the model and train it.
6. Fine-tune your model
7. Present your solution
8. Launch, monitor and maintain your system.

I followed all these steps in the jupyter notebook and I clearly explained all the insights

### Insights Obtained

![California Housing Prices Advanced plot](/california-housing-prices-prediction/images/end_to_end_project_practice/california_housing_prices_advanced_plt.png)


I used several models for this predictor but I settled on a Random forest Regressor where I selected the 15 best features and and the best hyperparameters using a random search CV. I pickled the best model and the best SVM I made.


## Project 2: MNIST Digits Classifier

We would be using the MNIST Dataset here,  which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labelled with the digit it represents. The dataset is already split into a well stratified 60,000 training set and 10,000 test set.

The Dataset was fetched using SciKit-Learn OpenML function (fetch_openml("mnist_784", version=1))


Some Digits from the Dataset
![Some Digits](/mnist-digits-classification/images/classification/more_digits_plot.jpg)



[Link to Notebook](mnist-digits-classification\mnist_dataset_classifcation.ipynb)

I built different types of classifiers first
1. Binary Classifier for digit 5
2. Multclass classifier (using both OvA and OvO) using SVM SVC (Resorts to OvO by default), SGD Classfifier
3. Multilabel Clasifier using a KNN Clasiffier
4. MultiOutput- Multiclass classfier to remove noise from a noisy digit image
5. A Multiclass Classfier for the MNIST dataset with over 97% Accuracy using a KNN Classifier and a GridSearchCV to select the best hyperparameters
6. Performed Data Augmentation to the MNIST Dataset (Added Shifted images for every digit) then I trained another KNN Clasiffier. This gave the highest accuracy.

I used classification metrics like Cross Validation, Confusion matrix, Precision/Recall Tradeoff (Precision score, Recall score and threshold),ROC curve and roc auc score, Error analysis methods after plotting the confusion matrix.

### Some Plots

Precision vs Recall
![Precision vs Recall](/mnist-digits-classification/images/classification/precision_vs_recall_plot.jpg)


Recall vs Threshold
![Recall vs Threshold](/mnist-digits-classification/images/classification/precision_recall_vs_threshold_plot.jpg)


ROC curve
![ROC curve](/mnist-digits-classification/images/classification/roc_curve_comparison_plot.jpg)


Error Analysis (The Diagonal was removed, the brighter values shows the highest inaccuracies)
![Error Analysis](/mnist-digits-classification/images/classification/conf_matrix_errors_colour_plot.jpg)




## Project 3: Titanic Survival Predictor

[Link to Dataset](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/)

### **Problem Statement**: The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on. This is a Kaggle competition and the dataset has already been split into a Training and Test set. The Test set doesn't contain the Labels.

### Output: 
An CSV containing the predicted value of "True" or "False", representing the "Survived" Label in the dataset. This CSV can be submitted on Kaggle to get your accuracy score.

### Methodology
1. After peeking at the data, I discovered that **Age**, **Cabin** and **Embarked** have some missing data. I dropped **Cabin** entirely and replaced the missing data of **Age** with median age after careful analysis.
2. I preprocessed the numerical and categorical data seperately. I converted the categorical data into a numerical form using OneHotEncoder
3. I then trained an Random Forest Classifier using 100 estimators and evaluated it (I acheieved an accuracy of 81.6% on the test set)
4. Next I tried a Support Vector Classifier and after cross validation I achieved 82.6% accuracy .
5. I then engineered some features:  
    * Converted numerical attributes to categorical attributes: for example, different age groups had very different survival rates (see below), so I created an age bucket category and use it instead of the age.
    * have a special category for people traveling alone since only 30% of them survived (see below).
    * Replace **SibSp** and **Parch** with their sum.
6. Then I tuned my SVC hyperparameters using a GridSearch until I achieved a accuracy of 82.71%.
7. I will try other models, evaluate my engineered features and continue some tuning hyperparameters



## Project 4: Spam Message Classifier

(Almost complete...Finishing Touches)

### Methodology
1. Obtained the dataset from this [Root Directory](http://spamassassin.apache.org/old/publiccorpus/) , but these are the specific links to the [spam](http://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2) and [ham](http://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2) mails.
2. I parsed the Emails and obtained the text from the raw html files using regular expressions. (For a more robust solution, you can use BeautifulSoup Library).
3. I carried out Stemming using the Natural Language Toolkit Module (NLTK). 
4. I created transformers to extract extra features for example: EmailsToWordCounter and WordCounterToVector. Extracting words works for many written languages like English. A vocabulary was obtained and word occurences was compared to the vocabulary.
5. Transformed the entire dataset using a Transformer pipeline consisting of EmailsToWordCounter and WordCounterToVector.
6. I Trained a Logistic Regression model and upon evaluation on the Training set, I obtained an accuracy of 98.5%.
7. Evaluation on the test set gives: 
- Precision: 96.88%
- Recall: 97.89%
8. I also explored using the BeautifulSoup Library for better parsing stability.

## Project  5: Description of some Machine Learning Algorithms

This notebook is dedicated to exploring and describing different Machine learning algorithms. I will discuss the following here:
1. Linear Regression
2. Gradient Descent
3. Mini Batch Gradient Descent
4. Polynomial Regression
(Work in Progress)








