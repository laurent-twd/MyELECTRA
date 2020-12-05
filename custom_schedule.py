import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate = 5e-4, weight_decay = 0.01, warmup_steps = 10000):
    super(CustomSchedule, self).__init__()
    
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.warmup_steps = tf.cast(warmup_steps, dtype = tf.float32)
    
  def __call__(self, step):
    
    warming_rate = self.learning_rate * step / self.warmup_steps 
    decaying_rate = - self.weight_decay * (step - self.warmup_steps) + self.learning_rate

    return tf.minimum(warming_rate, decaying_rate)

self = CustomSchedule()


step = tf.cast(tf.linspace(0, 100000, 100001), dtype = tf.float32)
y = tf.minimum(warming_rate, decaying_rate)
import matplotlib.pyplot as plt

p = tf.cast(step <= self.warmup_steps, dtype = tf.float32)

rate = p * warming_rate + (1-p) * decaying_rate
plt.plot(step, rate)
