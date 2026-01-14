import tensorflow as tf

model = tf.keras.models.load_model("kws_model")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("kws_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported kws_model.tflite")
