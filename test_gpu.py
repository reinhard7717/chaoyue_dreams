import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU已成功启用")
    except RuntimeError as e:
        print(e)
else:
    print("未检测到GPU，请检查配置")