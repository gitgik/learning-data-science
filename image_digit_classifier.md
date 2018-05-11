

```python
# train a model to classify handwritten digits
from keras.datasets import mnist
from keras import models, layers

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

print(training_images.shape)
```

    (60000, 28, 28)



```python
# create a network with two fully-contencted(dense) layers.
# the 2nd layer return an array of 10 probability scores summing to 1.
# each score is the probability that the current digit image belongs to one of the 10 digit classes.

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

```


```python
# compile the network by specifying the loss function, the optimizer and metrics to monitor during training
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
# preprocess the data reshaping it into the shape the network expects
# and scaling it so that all values are in [0, 1] interval
training_images = training_images.reshape((60000, 28 * 28))
training_images = training_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255




```


```python
from keras.utils import to_categorical
# encode the labels
training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)
print (test_labels.shape)
```

    (10000, 10)



```python
# train the model
network.fit(training_images, training_labels, epochs=5, batch_size=128)
```

    Epoch 1/5
    60000/60000 [==============================] - 4s 73us/step - loss: 0.2528 - acc: 0.9268
    Epoch 2/5
    60000/60000 [==============================] - 4s 71us/step - loss: 0.1037 - acc: 0.9698
    Epoch 3/5
    60000/60000 [==============================] - 4s 71us/step - loss: 0.0680 - acc: 0.9791
    Epoch 4/5
    60000/60000 [==============================] - 4s 72us/step - loss: 0.0498 - acc: 0.9847
    Epoch 5/5
    60000/60000 [==============================] - 4s 72us/step - loss: 0.0370 - acc: 0.9887





    <keras.callbacks.History at 0x1234fe4a8>




```python
# evaluate the accuracy of the model
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("{}%".format(test_accuracy * 100))
```

    10000/10000 [==============================] - 0s 50us/step
    98.22%

