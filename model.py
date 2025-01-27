import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from subprocess import call
 
data = pd.read_csv('abalone.data', sep=',', header=None)
data[0] = data[0].replace({'M': 0, 'F': 1, 'I': 2})
data[8] = data[8].replace({range(1,8): 1, range(8,11): 2, range(11,16): 3, range(16,30): 4})
Features = ["Sex","Length","Diameter","Height","Whole Weight","Shucked Weight","Viscera Weight","Shell Weight"]
Target = ["1-7","8-10","11-15", "15+"]

corr_matrix = data.corr()
print(corr_matrix)

plt.figure(figsize=(8, 6))
data[8].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribution of Classes")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig("Distribution of Classes.png", dpi=100)
plt.clf()

class_values = [1, 2, 3, 4]
plt.figure(figsize=(12, 8))
for i, class_value in enumerate(class_values):
    plt.subplot(2, 2, i + 1)
    plt.hist(data[data[8] == class_value][7], color='red', edgecolor="black")
    plt.title(f'Histogram for Class {class_value}')
    plt.xlabel('Shell Weight')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("Distribution of Shell Weight by Class.png", dpi=100)
plt.clf()

count_data = data.groupby([0, 8]).size().unstack()
plt.figure(figsize=(12, 8))
count_data.plot(kind='bar', stacked=True)
plt.title("Distribution of Sex by Class")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Class (Age)", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("Distribution of Sex by Class.png", dpi=100)
plt.clf()

X = data.iloc[:,0:8]
Y = data[8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=99, test_size = 0.4)

Full_tree = DecisionTreeClassifier(random_state=99)
Full_tree= Full_tree.fit(X_train, Y_train)

#r = export_text(Full_tree, feature_names= Features)
#print(r)

#export_graphviz(Full_tree, out_file='Full.dot', 
#                feature_names = Features,
#                class_names = Target,
#                rounded = True, proportion = False, 
#                precision = 2, filled = True)

#call(['dot', '-Tpng', 'Full.dot', '-o', 'Full.png', '-Gdpi=100'])

#plt.figure(figsize = (14, 18))
#plt.imshow(plt.imread('Full.png'))
#plt.axis('off')
#plt.show()
#plt.clf()

path = Full_tree.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig("Total Impurity vs effective alpha for training set.png", dpi=100)
plt.clf()

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=99, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.savefig("Number of nodes vs alpha.png", dpi=100)
plt.clf()

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

max_test_accuracy = max(test_scores)
best_ccp_alpha = ccp_alphas[test_scores.index(max_test_accuracy)]
print("Maximum Test Accuracy:", max_test_accuracy)
print("Best ccp_alpha value:", best_ccp_alpha)

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig("Accuracy vs alpha for training and testing sets.png", dpi=100)
plt.clf()

Pruned_tree = DecisionTreeClassifier(random_state=99, ccp_alpha= best_ccp_alpha)
Pruned_tree = Pruned_tree.fit(X_test,Y_test)
#export_graphviz(Pruned_tree, out_file='Pruned.dot', 
#                feature_names = Features,
#                class_names = Target,
#                rounded = True, proportion = False, 
#                precision = 2, filled = True)

#call(['dot', '-Tpng', 'Pruned.dot', '-o', 'Pruned.png', '-Gdpi=100'])

#plt.figure(figsize = (14, 18))
#plt.imshow(plt.imread('Pruned.png'))
#plt.axis('off')
#plt.show()
#plt.clf()

n_estimators_values = range(1, 11)
train_scores_forest = []
test_scores_forest = []

for n_estimators in n_estimators_values:
    Forest = RandomForestClassifier(n_estimators=n_estimators, random_state=99)
    Forest.fit(X_train, Y_train)
    
    train_score = Forest.score(X_train, Y_train)
    test_score = Forest.score(X_test, Y_test)
    
    train_scores_forest.append(train_score)
    test_scores_forest.append(test_score)

fig, ax = plt.subplots()
ax.set_xlabel("n_estimators")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs n_estimators for training and testing sets")
ax.plot(n_estimators_values, train_scores_forest, marker="o", label="train", drawstyle="steps-post")
ax.plot(n_estimators_values, test_scores_forest, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig("Accuracy vs n_estimators for training and testing sets.png", dpi=100)
plt.clf()

train_scores_forest_pruned = []
test_scores_forest_pruned = []

for n_estimators in n_estimators_values:
    Forest_pruned = RandomForestClassifier(n_estimators=n_estimators, random_state=99, ccp_alpha=best_ccp_alpha)
    Forest_pruned.fit(X_train, Y_train)
    
    train_score = Forest_pruned.score(X_train, Y_train)
    test_score = Forest_pruned.score(X_test, Y_test)
    
    train_scores_forest_pruned.append(train_score)
    test_scores_forest_pruned.append(test_score)

fig, ax = plt.subplots()
ax.set_xlabel("n_estimators")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs n_estimators for training and testing sets with Pruned Trees")
ax.plot(n_estimators_values, train_scores_forest_pruned, marker="o", label="train", drawstyle="steps-post")
ax.plot(n_estimators_values, test_scores_forest_pruned, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig("Accuracy vs n_estimators for training and testing sets with Pruned Trees.png", dpi=100)
plt.clf()

Forest_n10 = RandomForestClassifier(n_estimators=10, random_state=99)
Forest_n10.fit(X_train, Y_train)
Forest_pruned_n10 = RandomForestClassifier(n_estimators=10, random_state=99, ccp_alpha=best_ccp_alpha)
Forest_pruned_n10.fit(X_train, Y_train)

Full_tree_train_accuracy = Full_tree.score(X_train, Y_train)
Full_tree_test_accuracy = Full_tree.score(X_test, Y_test)
Pruned_tree_train_accuracy = Pruned_tree.score(X_train, Y_train)
Pruned_tree_test_accuracy = Pruned_tree.score(X_test, Y_test)
Forest_n10_train_accuracy = Forest_n10.score(X_train, Y_train)
Forest_n10_test_accuracy = Forest_n10.score(X_test, Y_test)
Forest_pruned_n10_train_accuracy = Forest_pruned_n10.score(X_train, Y_train)
Forest_pruned_n10_test_accuracy = Forest_pruned_n10.score(X_test, Y_test)

print("Full Tree Train Accuracy:", Full_tree_train_accuracy)
print("Full Tree Test Accuracy:", Full_tree_test_accuracy)
print("Pruned Tree Train Accuracy:", Pruned_tree_train_accuracy)
print("Pruned Tree Test Accuracy:", Pruned_tree_test_accuracy)
print("Random Forest Train Accuracy:", Forest_n10_train_accuracy)
print("Random Forest Test Accuracy:", Forest_n10_test_accuracy)
print("Pruned Random Forest Train Accuracy:", Forest_pruned_n10_train_accuracy)
print("Pruned Random Forest Test Accuracy:", Forest_pruned_n10_test_accuracy)

