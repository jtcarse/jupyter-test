

```python
import tensorflow as tf
```


```python
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
```


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```


```python
model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test, y_test)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 3s 42us/sample - loss: 0.3019 - accuracy: 0.9128
    Epoch 2/5
    60000/60000 [==============================] - 2s 40us/sample - loss: 0.1470 - accuracy: 0.9564
    Epoch 3/5
    60000/60000 [==============================] - 2s 40us/sample - loss: 0.1085 - accuracy: 0.9671
    Epoch 4/5
    60000/60000 [==============================] - 3s 45us/sample - loss: 0.0894 - accuracy: 0.9721
    Epoch 5/5
    60000/60000 [==============================] - 3s 52us/sample - loss: 0.0775 - accuracy: 0.9761
    10000/10000 [==============================] - 0s 32us/sample - loss: 0.0764 - accuracy: 0.9777





    [0.07638132400300819, 0.9777]


