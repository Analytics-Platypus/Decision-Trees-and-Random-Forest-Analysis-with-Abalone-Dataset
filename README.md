This Python script processes and analyzes the abalone dataset using decision trees and random forests. The objective is to classify abalone age into specific ranges based on physical and biological features. Below is a step-by-step explanation of the script's execution:

1. Dataset Preparation
The script uses the abalone.data file, which is loaded using Pandas.
Feature Sex (column 0) is encoded as integers (M=0, F=1, I=2).
The target variable (column 8) is mapped into four classes:
1: Age 1-7
2: Age 8-10
3: Age 11-15
4: Age 16+

2. Exploratory Data Analysis (EDA)
A correlation matrix is computed to understand relationships between features.
Several plots are generated for EDA:
Class Distribution: Bar chart showing the distribution of the four age classes.
Shell Weight Distribution by Class: Histograms for each class.
Sex Distribution by Class: Stacked bar chart showing class distributions by sex.
The plots are saved as PNG files for visual analysis.

3. Splitting the Dataset
The dataset is split into training (60%) and testing (40%) sets using train_test_split.

4. Building and Visualizing Decision Trees
Full Decision Tree:
A decision tree is trained on the training data using DecisionTreeClassifier.
A complexity pruning path is computed to analyze tree performance with different ccp_alpha values:
Plots are generated for:
Total impurity vs. ccp_alpha.
Number of nodes and tree depth vs. ccp_alpha.
Pruned Decision Tree:

The best ccp_alpha value is selected based on maximum test accuracy.
A pruned decision tree is trained using this value.

5. Random Forest Implementation
Random forests are built with varying numbers of estimators (n_estimators from 1 to 10):
Both unpruned and pruned versions are trained.
Accuracy vs. n_estimators plots are created for both training and testing sets.
Final models are trained with 10 estimators:
A regular random forest.
A pruned random forest using the best ccp_alpha value.

6. Performance Evaluation
Accuracy scores are calculated and displayed for:
Full decision tree.
Pruned decision tree.
Random forest (unpruned).
Pruned random forest.
Metrics include both training and testing accuracies.

7. Generated Output
The script outputs:

Visualizations for data exploration and model performance:
Class distributions, feature histograms, and impurity vs. ccp_alpha plots.
Accuracy and complexity plots for decision trees and random forests.
Model performance metrics printed to the console.

Prerequisites
Ensure the following Python packages are installed:
pandas
numpy
matplotlib
sklearn
Run the script in an environment with access to the abalone.data file.
