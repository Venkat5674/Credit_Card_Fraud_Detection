# Step 1: Importing all the necessary Libraries
We begin by importing the necessary Python libraries: numpy, pandas, matplotlib and seaborn for data handling, visualization and model building.

# Step 2: Loading the Data
Load the dataset into a pandas DataFrame and examine its structure

Dataset Download Link [https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv]

The dataset contains 284,807 transactions with 31 features including:

Time: This shows how many seconds have passed since the first transaction in the dataset.

V1-V28: These are special features created to hide sensitive information about the original data.

Amount: Transaction amount.

Class: Target variable (0 for normal transactions, 1 for fraudulent transactions).

# Step 3: Analyze Class Distribution
The next step is to check the distribution of fraudulent vs. normal transactions.

Explanation:

This code separates the dataset into two groups: fraudulent transactions (Class == 1) and valid transactions (Class == 0).

It calculates the ratio of fraud cases to valid cases to measure how imbalanced the dataset is.

It then prints the outlier fraction along with the number of fraud and valid transactions.

This analysis is crucial in fraud detection, as it reveals how rare fraud cases are and whether techniques like resampling or special evaluation metrics are needed.

Since the dataset is highly imbalanced with only 0.02% fraudulent transactions. we’ll first try to build a model without balancing the dataset. If we don’t get satisfactory results we will explore ways to handle the imbalance.

# Step 4: Visualize Transaction Amounts
Let's compare the transaction amounts for fraudulent and normal transactions. This will help us understand if there are any significant differences in the monetary value of fraudulent transactions.

# Step 5: Correlation Matrix
We can visualize the correlation between features using a heatmap using correlation matrix. This will give us an understanding of how the different features are correlated and which ones may be more relevant for prediction.

# Step 6: Preparing Data
Separate the input features (X) and target variable (Y) then split the data into training and testing sets

Explanation:

X = data.drop(['Class'], axis = 1) removes the target column (Class) from the dataset to keep only the input features.

Y = data["Class"] selects the Class column as the target variable (fraud or not).

X.shape and Y.shape print the number of rows and columns in the feature set and the target set.

xData = X.values and yData = Y.values convert the Pandas DataFrame/Series to NumPy arrays for faster processing.

train_test_split(...) splits the data into training and testing sets:

80% for training, 20% for testing.

random_state=42 ensures reproducibility (same split every time you run it).


# Step 7: Building and Training Model
Train a Random Forest Classifier to predict fraudulent transactions.

Explanation:

from sklearn.ensemble import RandomForestClassifier: This imports the RandomForestClassifier from sklearn.ensemble, which is used to create a random forest model for classification tasks.

rfc = RandomForestClassifier(): Initializes a new instance of the RandomForestClassifier.

rfc.fit(xTrain, yTrain): Trains the RandomForestClassifier model on the training data (xTrain for features and yTrain for the target labels).

yPred = rfc.predict(xTest): Uses the trained model to predict the target labels for the test data (xTest), storing the results in yPred.

# Step 8: Handling Missing Values
Before proceeding with model evaluation it’s important to ensure that the dataset does not contain any missing or invalid values e.g, Nan. 

If your dataset contains missing values in the target variable yTest you can use the SimpleImputer from sklearn.impute to handle them.

Explanation:

print("NaN values in yTest before imputation:", np.isnan(yTest).sum()): Prints the number of NaN values in yTest before imputation.

imputer = SimpleImputer(strategy='most_frequent'): Initializes the imputer to replace NaN values with the most frequent value.

yTest = imputer.fit_transform(yTest.reshape(-1, 1)).flatten(): Reshapes yTest, applies imputation, and flattens it back to 1D.

# Step 9: Evaluating the Model

After training the model we need to evaluate its performance using various metrics such as accuracy, precision, recall, F1-score and the Matthews correlation coefficient.

# Model Evaluation Metrics:
The model accuracy is high due to class imbalance so we will have computed precision, recall and f1 score to get a more meaningful understanding. We observe:

Accuracy: 0.9996: Out of all predictions, 99.96% were correct. However, in imbalanced datasets (like fraud detection), accuracy can be misleading — a model that predicts everything as "not fraud" will still have high accuracy.

Precision: 0.9873: When the model predicted "fraud", it was correct 98.73% of the time. High precision means very few false alarms (false positives).

Recall: 0.7959: Out of all actual fraud cases, the model detected 79.59%. This shows how well it catches real frauds. A lower recall means some frauds were missed (false negatives).

F1-Score: 0.8814: A balance between precision and recall. 88.14% is strong and shows the model handles both catching fraud and avoiding false alarms well.

Matthews Correlation Coefficient (MCC): 0.8863: A more balanced score (from -1 to +1) even when classes are imbalanced. A value of 0.8863 is very good, it means the model is making strong, balanced predictions overall.
