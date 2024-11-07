import tensorflow as tf

w = tf.Variable(3.0)
b = tf.Variable(2.0)
x = 1.0
y = 1.0
alpha = 0.1

iterations = 30

optimizers = tf.keras.optimizers.Adam(learning_rate=0.1)

for iter in range(iterations):
    with tf.GradientTape() as tape:
        fwb = w * x + b
        costJ = (fwb - y) ** 2
        
    grads = tape.gradient(costJ, [w, b])
    optimizers.apply_gradients(zip(grads, [w, b]))
    
print(w, b)