# Machine-Learning-Projects

A Repo for my AI, Data Science and Machine Learning Projects


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

![California Housing Prices Advanced plot](https://github.com/uzoochogu/Machine-Learning-Projects/blob/main/california-housing-prices-prediction/images/end_to_end_project_practice/california_housing_prices_advanced_plt.png)


I used several models for this predictor but I settled on a Random forest Regressor where I selected the 15 best features and and the best hyperparameters using a random search CV. I pickled the best model and the best SVM I made.


## Project 2: MNIST Digits Classifier

We would be using the MNIST Dataset here,  which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labelled with the digit it represents. The dataset is already split into a well stratified 60,000 training set and 10,000 test set.

The Dataset was fetched using SciKit-Learn OpenML function (fetch_openml("mnist_784", version=1))


Some Digits from the Dataset
![Some Digits](https://github.com/uzoochogu/Machine-Learning-Projects/blob/main/mnist-digits-classification/images/classification/more_digits_plot.png)



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
![Precision vs Recall](https://github.com/uzoochogu/Machine-Learning-Projects/blob/main/mnist-digits-classification/images/classification/precision_vs_recall_plot.png)


Recall vs Threshold
![Recall vs Threshold](https://github.com/uzoochogu/Machine-Learning-Projects/blob/main/mnist-digits-classification/images/classification/precision_recall_vs_threshold_plot.png)


ROC curve
![ROC curve](https://github.com/uzoochogu/Machine-Learning-Projects/blob/main/mnist-digits-classification/images/classification/roc_curve_comparison_plot.png)


Error Analysis (The Diagonal was removed, the brighter values shows the highest inaccuracies)
![Error Analysis](https://github.com/uzoochogu/Machine-Learning-Projects/blob/main/mnist-digits-classification/images/classification/conf_matrix_errors_colour_plot.png)




