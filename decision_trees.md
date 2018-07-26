
#### Entropy
In order to understand decision trees, we have to learn about entropy. 
Entropy is the degree of randomness or disorder in the system.

Let's use water ice and vapour to explain this concept. 

Ice has a low entropy because particles cannot move around at all, water has some particles moving around so it's medium entropy. Vapour has a higher entropy because its particles have the freedom to move around faster and further.

So the more rigid(homogeneous) the set, the less entropy it will have and vice versa. Also the more knowledge one has on a set, the less entropy and vice versa. A set that has different entities has much more entropy than a set containing an entity of the same nature.

 #### Hyperparameters for Decision Trees
For us to create decision trees that will generalize well, we need to tune some aspects of the trees. These aspects are called hyperparameters.

Here are the most important ones:
1. Maximum Depth –– This is the largest length between the root and a leaf. A tree with length $k$ can have at most $2^k$ leaves
2. Minumum number of samples per leaf –– To prevent instances like having 99% of the samples being in a single leaf and only 1% on the other, we can set a minimum for the number of sample for each leaf.
3. Maximum number of features –– most times we have too many features to build a tree. On every split, it would be expensive to check the entire dataset on each of the features. A solution for this is to limit the number of features that we look for in each split.

Points to note:
* Large depths can cause overfitting –– a tree that is too deep can memorize data. we don't want that.
* Small depths can cause underfitting –– a tree that is too small can result in a very simple model.
* Small minimum samples per leaf –– results in leafs with very small samples which results in the tree memorizing data/overfitting
* Large minimum samples per lead –– results in leafs that have very large samples which lead to smaller underfitted models




#### Let's fit a decision tree!
Tools we need to create a decision tree
* Scikit-learn –– we'll create a decision tree model using Scikit-learn's `Decision Tree Classifier` class.
This class provides a way to fit our model to the data.


#### 1. Load the data
The data we'll be using is in a data.csv file. It has three columns, the first 2 contains cordinates of the points, and the third one of the label.

We'll then split the data intro features `X` and labels `y`

    


```python
# read the data
data = np.asarray(pd.read_csv('decision-tree-data.csv', header=None))
# assign features to variable X, and the labels to variable y
X = data[:, 0:2]
y = data[:,2]
```

#### 2. Build the decision tree model
When we define the model, we can specify hyperparameters such as:
* `max_depth` –– maximum number of levels in the tree
* `min_samples_leaf` –– minimum number of samples allowed in a leaf
* `min_samples_split` –– minimum number of samples required to split an internal node.
* `max_features` –– number of features to consider when looking for the best split.


```python
model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=1, max_features=2)
```

#### 3. Fit the model to the data


```python
model.fit(X, y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
                max_features=2, max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')



#### 4. Predict using the model



```python
# make predictions
y_predict = model.predict(X)
```

#### 5. Calculate the accuracy of the model
We'll use sklearn's function `accuracy_score()` to measure the accuracy metric. 

This process will help us finetune the hyperparameters to get an accuracy of 100% on the set. 

Although very large values for the parameters will fit the training set quite well, we should avoid them for they will do so by overfitting the model. 

We should rather find the smallest possible values that have a less chance of overfitting.


```python
# test the accuracy of the model
score = accuracy_score(y, y_predict)
print("Accuracy score is: {}%".format(score * 100))

```

    Accuracy score is: 100.0%

