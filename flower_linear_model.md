
#### A Deep Neural Network (DNN) model to discern Iris flowers
We'll build a DNN model to predict the type of Iris flower species given only a flower's natural features(sepal and petal length).

We will use Google's Tensorflow: an open source machine learning tool for everyone.

The types of flowers we would like to distunguish is Iris Versicolour, Iris Virginica, Iris Setosa. 

The flowers look like this:

* Iris Virginica
![Iris Virginica](images/178px-Iris_virginica.jpg)

* Iris Versicolor
![Iris_versicolor](images/193px-Iris_versicolor_3.jpg)

* Iris Setosa
![Iris Kosaciec](images/109px-Kosaciec_szczecinkowaty_Iris_setosa.jpg)

The data columns we have are:
* sepal length in cm
* sepal width in cm
* petal length in cm
* petal width in cm

These data points will be used to train the model and test its accuracy.

Time to write some Neural Nets!



```python
# Step 1: split the data into training and testing data
# Key points to note:
############ In the data frame that we wish to form,
############ * Setosa's value = 0, Versicolor = 1 and Virginica = 2
########### * The species column will be changed to contain these values before we split the data.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# read data from the csv file
actual_data = pd.read_csv("iris.csv")
data = pd.read_csv("iris.csv", skiprows=[0], header=None)
labels_data = pd.read_csv("iris.csv", usecols=[4], skiprows=[0], header=None)
labels = np.unique(np.array(labels_data, 'str'))
print (labels)

# iterate through each row, changing the last column (4) to have integers instead of flower names
for index, row in data.iterrows():
    if row[4] == labels[0]:
        data.loc[index, 4] = 0
    elif row[4] == labels[1]:
        data.loc[index, 4] = 1
    else:
        data.loc[index, 4] = 2

# shuffle the newly formatted data
data = data.sample(frac=1).reset_index(drop=True)
y = actual_data.species

# split that data, 80-20 rule
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print (X_test.head())

# write the split data to csv
# but first create a custom header for the training and test csv files
# and omit the species column from the header since that's what we want to predict (hence the "- 1")
X_train_header = list((len(X_train.index), len(X_train.columns) - 1))  + list(labels)
X_test_header = list((len(X_test.index), len(X_test.columns) - 1)) + list(labels)
print (X_train_header)
print (X_test_header)

# write the split data to csv files
X_train.to_csv("iris_training.csv", index=False, index_label=False, header=X_train_header)
X_test.to_csv("iris_test.csv", index=False, index_label=False, header=X_test_header)
```

    ['setosa' 'versicolor' 'virginica']
           0    1    2    3  4
    48   6.0  2.2  5.0  1.5  2
    132  6.2  2.2  4.5  1.5  1
    107  5.7  2.8  4.5  1.3  1
    3    5.6  2.9  3.6  1.3  1
    100  6.3  2.7  4.9  1.8  2
    [120, 4, 'setosa', 'versicolor', 'virginica']
    [30, 4, 'setosa', 'versicolor', 'virginica']


The reason why we had to format the header of the training and test csv to be 
`(row length, column length, 'setosa', 'versicolor', 'virginica')` is to 
make the `load_csv_with_header` function be able to read our data. Data preparation matters. :-)


```python
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

# Data files containing our sepal and petal features
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

training_set = base.load_csv_with_header(
    filename=IRIS_TRAINING, 
    features_dtype=np.float32, 
    target_dtype=np.int)

test_set = base.load_csv_with_header(
    filename=IRIS_TEST,
    features_dtype=np.float32,
    target_dtype=np.int)

print(test_set.target)



```

    [2 1 1 1 2 0 0 1 1 2 0 0 0 2 0 0 1 2 0 0 0 2 1 1 1 1 0 0 2 1]



```python
# Specify that all feature columns have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
```

Now, we build the 3 layer Deep Nueral Network classfier.


```python
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="/tmp/iris_model"
)
```

    WARNING:tensorflow:From /Users/jee/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/estimators/dnn.py:378: multi_class_head (from tensorflow.contrib.learn.python.learn.estimators.head) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please switch to tf.contrib.estimator.*_head.
    WARNING:tensorflow:From /Users/jee/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/estimators/estimator.py:1165: BaseEstimator.__init__ (from tensorflow.contrib.learn.python.learn.estimators.estimator) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please replace uses of any Estimator from tf.contrib.learn with an Estimator from tf.estimator.*
    WARNING:tensorflow:From /Users/jee/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/estimators/estimator.py:427: RunConfig.__init__ (from tensorflow.contrib.learn.python.learn.estimators.run_config) is deprecated and will be removed in a future version.
    Instructions for updating:
    When switching to tf.estimator.Estimator, use tf.estimator.RunConfig instead.
    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_task_type': None, '_task_id': 0, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1c2583e470>, '_master': '', '_num_ps_replicas': 0, '_num_worker_replicas': 0, '_environment': 'local', '_is_chief': True, '_evaluation_master': '', '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1.0
    }
    , '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_secs': 600, '_log_step_count_steps': 100, '_session_config': None, '_save_checkpoints_steps': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': '/tmp/iris_model'}


Next, we define the training inputs


```python
def training_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    
    return x,y
```

Then next step is to fit the model with the training data. Fitting is where the model is trained.


```python
# fit the classifier with the training data
classifier.fit(input_fn=training_inputs, steps=2000)
    
```

    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-14000
    INFO:tensorflow:Saving checkpoints for 14001 into /tmp/iris_model/model.ckpt.
    INFO:tensorflow:loss = 0.04674709, step = 14001
    INFO:tensorflow:global_step/sec: 783.448
    INFO:tensorflow:loss = 0.046954535, step = 14101 (0.131 sec)
    INFO:tensorflow:global_step/sec: 860.518
    INFO:tensorflow:loss = 0.047011122, step = 14201 (0.115 sec)
    INFO:tensorflow:global_step/sec: 890.17
    INFO:tensorflow:loss = 0.046789907, step = 14301 (0.111 sec)
    INFO:tensorflow:global_step/sec: 637.91
    INFO:tensorflow:loss = 0.046695717, step = 14401 (0.159 sec)
    INFO:tensorflow:global_step/sec: 778.683
    INFO:tensorflow:loss = 0.04667021, step = 14501 (0.127 sec)
    INFO:tensorflow:global_step/sec: 735.148
    INFO:tensorflow:loss = 0.046662346, step = 14601 (0.141 sec)
    INFO:tensorflow:global_step/sec: 890.48
    INFO:tensorflow:loss = 0.046659503, step = 14701 (0.108 sec)
    INFO:tensorflow:global_step/sec: 678.196
    INFO:tensorflow:loss = 0.0466594, step = 14801 (0.150 sec)
    INFO:tensorflow:global_step/sec: 724.953
    INFO:tensorflow:loss = 0.04666304, step = 14901 (0.135 sec)
    INFO:tensorflow:global_step/sec: 789.559
    INFO:tensorflow:loss = 0.046675432, step = 15001 (0.131 sec)
    INFO:tensorflow:global_step/sec: 579.811
    INFO:tensorflow:loss = 0.04671428, step = 15101 (0.169 sec)
    INFO:tensorflow:global_step/sec: 680.94
    INFO:tensorflow:loss = 0.04681112, step = 15201 (0.145 sec)
    INFO:tensorflow:global_step/sec: 936.521
    INFO:tensorflow:loss = 0.04689629, step = 15301 (0.109 sec)
    INFO:tensorflow:global_step/sec: 883.049
    INFO:tensorflow:loss = 0.0468172, step = 15401 (0.111 sec)
    INFO:tensorflow:global_step/sec: 762.016
    INFO:tensorflow:loss = 0.04672064, step = 15501 (0.131 sec)
    INFO:tensorflow:global_step/sec: 755.378
    INFO:tensorflow:loss = 0.046678133, step = 15601 (0.136 sec)
    INFO:tensorflow:global_step/sec: 728.151
    INFO:tensorflow:loss = 0.0466625, step = 15701 (0.134 sec)
    INFO:tensorflow:global_step/sec: 746.197
    INFO:tensorflow:loss = 0.046656676, step = 15801 (0.134 sec)
    INFO:tensorflow:global_step/sec: 680.698
    INFO:tensorflow:loss = 0.046654765, step = 15901 (0.151 sec)
    INFO:tensorflow:Saving checkpoints for 16000 into /tmp/iris_model/model.ckpt.
    INFO:tensorflow:Loss for final step: 0.046655238.





    DNNClassifier(params={'head': <tensorflow.contrib.learn.python.learn.estimators.head._MultiClassHead object at 0x1c21f97358>, 'hidden_units': [10, 20, 10], 'feature_columns': (_RealValuedColumn(column_name='', dimension=4, default_value=None, dtype=tf.float32, normalizer=None),), 'optimizer': None, 'activation_fn': <function relu at 0x1160d4d90>, 'dropout': None, 'gradient_clip_norm': None, 'embedding_lr_multipliers': None, 'input_layer_min_slice_size': None})




```python
# Define test inputs
def test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    
    return x, y
```

After training, we evaluate the accuracy of our trained model. 
We do this using the evaluate method. It takes in the test input data and target to build its input data pipeline. After measuring the model's accuracy, it returns a dictionary containing the results.


```python
# evaluate the classifier's accuracy
accuracy_score = classifier.evaluate(input_fn=test_inputs, steps=1)['accuracy']
print ("Accuracy score", accuracy_score)
```

    INFO:tensorflow:Starting evaluation at 2018-03-05-11:23:42
    INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-10000
    INFO:tensorflow:Evaluation [1/1]
    INFO:tensorflow:Finished evaluation at 2018-03-05-11:23:43
    INFO:tensorflow:Saving dict for global step 10000: accuracy = 1.0, global_step = 10000, loss = 0.015831133
    Accuracy score 1.0


Time to see if our model can predict the type of Iris flower given a new flower sample. 


```python
# Classify two new flower samples.
def new_flower_samples():
    return np.array(
        [[6.4, 3.2, 4.5, 1.5],
        [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

```


```python
# Predict the type of Iris flower
prediction = classifier.predict_classes(input_fn=new_flower_samples)
print (list(prediction))
```

    INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-10000
    [1, 2]


We made it!!

Now, let's try creating a Linear model and compare it's prediction with that of the DNN we just created


```python
# Create a built in linear model classifier
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]

# Create a classifier that takes in the feature columns above, specify the number of outputs to predict ==3
# and the model directory to store the model's training progress and the output files
linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                                 n_classes=3,
                                                 model_dir="/tmp/iris-linear-model")

```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': '/tmp/iris-linear-model', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1c20855ef0>, '_task_type': 'worker', '_task_id': 0, '_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
# fit the linear model with training data
def input_fn(dataset):
    def _fn():
        features = {feature_name: dataset.data}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

linear_classifier.train(input_fn=input_fn(training_set), steps=1000)
```

    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Saving checkpoints for 1 into /tmp/iris-linear-model/model.ckpt.
    INFO:tensorflow:loss = 131.83344, step = 1
    INFO:tensorflow:global_step/sec: 734.543
    INFO:tensorflow:loss = 37.549465, step = 101 (0.138 sec)
    INFO:tensorflow:global_step/sec: 839.511
    INFO:tensorflow:loss = 28.260448, step = 201 (0.118 sec)
    INFO:tensorflow:global_step/sec: 772.602
    INFO:tensorflow:loss = 23.427868, step = 301 (0.130 sec)
    INFO:tensorflow:global_step/sec: 807.648
    INFO:tensorflow:loss = 20.424622, step = 401 (0.127 sec)
    INFO:tensorflow:global_step/sec: 1016.51
    INFO:tensorflow:loss = 18.36332, step = 501 (0.099 sec)
    INFO:tensorflow:global_step/sec: 987.209
    INFO:tensorflow:loss = 16.853485, step = 601 (0.101 sec)
    INFO:tensorflow:global_step/sec: 584.139
    INFO:tensorflow:loss = 15.695606, step = 701 (0.171 sec)
    INFO:tensorflow:global_step/sec: 603.879
    INFO:tensorflow:loss = 14.776791, step = 801 (0.167 sec)
    INFO:tensorflow:global_step/sec: 938.491
    INFO:tensorflow:loss = 14.028194, step = 901 (0.104 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into /tmp/iris-linear-model/model.ckpt.
    INFO:tensorflow:Loss for final step: 13.411062.





    <tensorflow.python.estimator.canned.linear.LinearClassifier at 0x1c21532f28>




```python
# test the accuracy
linear_accuracy = linear_classifier.evaluate(input_fn=input_fn(test_set), steps=100)['accuracy']
print ("Linear model accuracy score: ", linear_accuracy)
```

    INFO:tensorflow:Starting evaluation at 2018-03-05-12:53:09
    INFO:tensorflow:Restoring parameters from /tmp/iris-linear-model/model.ckpt-1000
    INFO:tensorflow:Evaluation [1/100]
    INFO:tensorflow:Evaluation [2/100]
    INFO:tensorflow:Evaluation [3/100]
    INFO:tensorflow:Evaluation [4/100]
    INFO:tensorflow:Evaluation [5/100]
    INFO:tensorflow:Evaluation [6/100]
    INFO:tensorflow:Evaluation [7/100]
    INFO:tensorflow:Evaluation [8/100]
    INFO:tensorflow:Evaluation [9/100]
    INFO:tensorflow:Evaluation [10/100]
    INFO:tensorflow:Evaluation [11/100]
    INFO:tensorflow:Evaluation [12/100]
    INFO:tensorflow:Evaluation [13/100]
    INFO:tensorflow:Evaluation [14/100]
    INFO:tensorflow:Evaluation [15/100]
    INFO:tensorflow:Evaluation [16/100]
    INFO:tensorflow:Evaluation [17/100]
    INFO:tensorflow:Evaluation [18/100]
    INFO:tensorflow:Evaluation [19/100]
    INFO:tensorflow:Evaluation [20/100]
    INFO:tensorflow:Evaluation [21/100]
    INFO:tensorflow:Evaluation [22/100]
    INFO:tensorflow:Evaluation [23/100]
    INFO:tensorflow:Evaluation [24/100]
    INFO:tensorflow:Evaluation [25/100]
    INFO:tensorflow:Evaluation [26/100]
    INFO:tensorflow:Evaluation [27/100]
    INFO:tensorflow:Evaluation [28/100]
    INFO:tensorflow:Evaluation [29/100]
    INFO:tensorflow:Evaluation [30/100]
    INFO:tensorflow:Evaluation [31/100]
    INFO:tensorflow:Evaluation [32/100]
    INFO:tensorflow:Evaluation [33/100]
    INFO:tensorflow:Evaluation [34/100]
    INFO:tensorflow:Evaluation [35/100]
    INFO:tensorflow:Evaluation [36/100]
    INFO:tensorflow:Evaluation [37/100]
    INFO:tensorflow:Evaluation [38/100]
    INFO:tensorflow:Evaluation [39/100]
    INFO:tensorflow:Evaluation [40/100]
    INFO:tensorflow:Evaluation [41/100]
    INFO:tensorflow:Evaluation [42/100]
    INFO:tensorflow:Evaluation [43/100]
    INFO:tensorflow:Evaluation [44/100]
    INFO:tensorflow:Evaluation [45/100]
    INFO:tensorflow:Evaluation [46/100]
    INFO:tensorflow:Evaluation [47/100]
    INFO:tensorflow:Evaluation [48/100]
    INFO:tensorflow:Evaluation [49/100]
    INFO:tensorflow:Evaluation [50/100]
    INFO:tensorflow:Evaluation [51/100]
    INFO:tensorflow:Evaluation [52/100]
    INFO:tensorflow:Evaluation [53/100]
    INFO:tensorflow:Evaluation [54/100]
    INFO:tensorflow:Evaluation [55/100]
    INFO:tensorflow:Evaluation [56/100]
    INFO:tensorflow:Evaluation [57/100]
    INFO:tensorflow:Evaluation [58/100]
    INFO:tensorflow:Evaluation [59/100]
    INFO:tensorflow:Evaluation [60/100]
    INFO:tensorflow:Evaluation [61/100]
    INFO:tensorflow:Evaluation [62/100]
    INFO:tensorflow:Evaluation [63/100]
    INFO:tensorflow:Evaluation [64/100]
    INFO:tensorflow:Evaluation [65/100]
    INFO:tensorflow:Evaluation [66/100]
    INFO:tensorflow:Evaluation [67/100]
    INFO:tensorflow:Evaluation [68/100]
    INFO:tensorflow:Evaluation [69/100]
    INFO:tensorflow:Evaluation [70/100]
    INFO:tensorflow:Evaluation [71/100]
    INFO:tensorflow:Evaluation [72/100]
    INFO:tensorflow:Evaluation [73/100]
    INFO:tensorflow:Evaluation [74/100]
    INFO:tensorflow:Evaluation [75/100]
    INFO:tensorflow:Evaluation [76/100]
    INFO:tensorflow:Evaluation [77/100]
    INFO:tensorflow:Evaluation [78/100]
    INFO:tensorflow:Evaluation [79/100]
    INFO:tensorflow:Evaluation [80/100]
    INFO:tensorflow:Evaluation [81/100]
    INFO:tensorflow:Evaluation [82/100]
    INFO:tensorflow:Evaluation [83/100]
    INFO:tensorflow:Evaluation [84/100]
    INFO:tensorflow:Evaluation [85/100]
    INFO:tensorflow:Evaluation [86/100]
    INFO:tensorflow:Evaluation [87/100]
    INFO:tensorflow:Evaluation [88/100]
    INFO:tensorflow:Evaluation [89/100]
    INFO:tensorflow:Evaluation [90/100]
    INFO:tensorflow:Evaluation [91/100]
    INFO:tensorflow:Evaluation [92/100]
    INFO:tensorflow:Evaluation [93/100]
    INFO:tensorflow:Evaluation [94/100]
    INFO:tensorflow:Evaluation [95/100]
    INFO:tensorflow:Evaluation [96/100]
    INFO:tensorflow:Evaluation [97/100]
    INFO:tensorflow:Evaluation [98/100]
    INFO:tensorflow:Evaluation [99/100]
    INFO:tensorflow:Evaluation [100/100]
    INFO:tensorflow:Finished evaluation at 2018-03-05-12:53:10
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 1.0, average_loss = 0.11253287, global_step = 1000, loss = 3.375986
    Linear model accuracy score:  1.0


#### Serverless Predictions at Scale
We are going to use scale our model on the cloud using Cloud Machine Learning Engine Prediction Service.



```python
# take a snapshot of the model and export it as a set of files that you can used elsewhere.
feature_fn = {"flower_features": tf.FixedLenFeature(shape=[4], dtype=np.float32)}
serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(featu)
linear_classifier.export_savedmodel(
    export_di_base="/tmp/iris_model" + "/export", 
    serving_input_receiver_fn=serving_fn)
```

    INFO:tensorflow:Restoring parameters from /tmp/iris-linear-model/model.ckpt-1000
    INFO:tensorflow:Assets added to graph.
    INFO:tensorflow:No assets to write.
    INFO:tensorflow:SavedModel written to: b"/tmp/iris_model/export/temp-b'1520255873'/saved_model.pb"





    b'/tmp/iris_model/export/1520255873'


