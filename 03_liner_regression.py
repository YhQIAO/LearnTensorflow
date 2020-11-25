import tensorflow as tf
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

# numpy 实现梯度下降算法
a, b = 0, 0
num_epoch = 10000
learning_rate = 5e-4
for e in range(num_epoch):
    y_pred = a * X + b
    grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
print(a, b)

# tensorflow 计算线性回归
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred-y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)