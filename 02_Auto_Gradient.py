import tensorflow as tf

# x = tf.Variable(initial_value=3.)
# with tf.GradientTape() as tape:
#     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
#     y = tf.square(x)
#
# y_grad = tape.gradient(y,x)
# print(y_grad)

X = tf.constant(
    [
        [1., 2.],
        [3., 4.]
    ]
)

y = tf.constant(
    [
        [1.], [2.]
    ]
)

w = tf.Variable(initial_value=[
    [1.], [2.]
])

b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(
        tf.square(tf.matmul(X, w) + b - y)
    )
w_grad, b_grad = tape.gradient(L, [w, b])
print(L, w_grad, b_grad)