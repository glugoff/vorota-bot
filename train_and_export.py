# train_and_export.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === 1. Загрузка данных ===
data_dir = r"D:\code\vorota"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# === 2. Создание модели ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Замораживаем базовую модель
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 3. Обучение ===
print("🚀 Обучение модели...")
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# === 4. Сохраняем в .h5 (опционально) ===
model.save("gate_model_new.h5")
print("✅ Модель сохранена как gate_model_new.h5")

# === 5. Конвертация в TFLite (ПРОСТОЙ, КАК В ПЕРВЫЙ РАЗ) ===
print("🔄 Конвертация в TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open("gate_model_v214.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Совместимая TFLite-модель сохранена как gate_model_v214.tflite")