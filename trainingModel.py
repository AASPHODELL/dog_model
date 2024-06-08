import cv2
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.optimizers import Adam

# Загрузка датасета 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

train_path = pathlib.Path("Images")
val_path = pathlib.Path("Test")

img_h = 224
img_w = 224

train_data = tf.keras.utils.image_dataset_from_directory(train_path)
val_data = tf.keras.utils.image_dataset_from_directory(val_path)
class_labels = dict(zip(train_data.class_names, range(len(train_data.class_names))))
num_labels = len(class_labels)

# Предобработка ихображений
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

augmenter = ImageDataGenerator(
    preprocessing_function=mobilenet_v2.preprocess_input,
    rotation_range=32,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

train_gen = augmenter.flow_from_directory(train_path,
                                          target_size=(img_h, img_w),
                                          color_mode="rgb",
                                          class_mode="categorical",
                                          batch_size=32,
                                          shuffle=True,
                                          seed=123)

val_gen = augmenter.flow_from_directory(val_path,
                                        target_size=(img_h, img_w),
                                        color_mode="rgb",
                                        class_mode="categorical",
                                        batch_size=32,
                                        shuffle=True,
                                        seed=123)

# Создание и настройка модели
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

base_model = MobileNetV2(
    input_shape=(img_h, img_w, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

base_model.trainable = False

input_tensor = base_model.input
x = Dense(128, activation='relu')(base_model.output)
x = Dense(128, activation='relu')(x)
output_tensor = Dense(num_labels, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

# Установка обратных вызовов и компиляция модели
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=2,
    verbose=1,
    restore_best_weights=True,
)
model_ckpt = ModelCheckpoint('best_model.keras',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True)

callbacks_list = [early_stop, model_ckpt]

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Предобработка тестовых данных
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

test_gen = augmenter.flow_from_directory(val_path,
                                         target_size=(224, 224),
                                         color_mode="rgb",
                                         class_mode="categorical",
                                         batch_size=32,
                                         shuffle=False)

# Обучение модели
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

hist = model.fit(
    train_gen, validation_data=val_gen,
    epochs=20,
    callbacks=callbacks_list)

# Оценка модели
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

(eval_loss, eval_acc) = model.evaluate(test_gen)

train_loss_vals = hist.history['loss']
val_loss_vals = hist.history['val_loss']
train_acc_vals = hist.history['accuracy']
val_acc_vals = hist.history['val_accuracy']

# Визуализация результатов
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss_vals, label='Training Loss')
plt.plot(val_loss_vals, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_vals, label='Training Accuracy')
plt.plot(val_acc_vals, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
