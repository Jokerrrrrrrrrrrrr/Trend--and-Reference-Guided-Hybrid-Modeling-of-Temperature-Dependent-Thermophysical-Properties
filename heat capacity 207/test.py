import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

print("TensorFlow version:", tf.__version__)
print("Import successful!")

# 测试简单的模型创建
inputs = Input(shape=(10,))
x = Dense(64, activation='relu')(inputs)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
print("Model created successfully!")