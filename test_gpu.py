import tensorflow as tf
import sys

print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)

# 检查可用的物理设备
physical_devices = tf.config.list_physical_devices()
print("Physical devices:", physical_devices)

# 检查可用的逻辑设备 (包括 GPU)
logical_devices = tf.config.list_logical_devices()
print("Logical devices:", logical_devices)

# 检查是否有 GPU 设备
gpu_available = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_available))

if gpu_available:
    print("GPU is available and detected by TensorFlow.")
    # 打印 GPU 名称
    for gpu in gpu_available:
        print("GPU Name:", gpu.name)
else:
    print("GPU is not available or not detected by TensorFlow.")

# 运行一个简单的计算来测试 GPU
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Simple matrix multiplication result (should run on GPU if available):")
        print(c.numpy())
except tf.errors.InvalidArgumentError as e:
    print("Could not run matrix multiplication on GPU. Error:", e)
    print("Falling back to CPU.")
    with tf.device('/CPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Simple matrix multiplication result (running on CPU):")
        print(c.numpy())

