import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =====================================================
# 1) DATASET VE PARAMETRELER
# =====================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10   # 🔥 Epoch 10'a düşürüldü

train_dir = r"C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\train"
val_dir   = r"C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\val"
test_dir  = r"C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\test"

# =====================================================
# 2) DATA AUGMENTATION & GENERATORS
# =====================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", train_gen.class_indices)

# =====================================================
# 3) MODEL MİMARİSİ (CNN)
# =====================================================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same",
           input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# 4) CALLBACKS
# =====================================================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# =====================================================
# 5) MODEL EĞİTİMİ
# =====================================================
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stopping, lr_scheduler]
)

# =====================================================
# 6) LOSS & ACCURACY GRAFİKLERİ
# =====================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Analysis")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Analysis")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =====================================================
# 7) OVERFITTING OTOMATİK ANALİZ
# =====================================================
final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]

if final_val_loss > final_train_loss * 1.2:
    print("⚠️ Overfitting ihtimali var.")
else:
    print("✅ Overfitting gözlemlenmedi.")

# =====================================================
# 8) TEST SET DEĞERLENDİRME
# =====================================================
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

y_prob = model.predict(test_gen).ravel()
y_pred = (y_prob >= 0.5).astype(int)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# =====================================================
# 9) CONFUSION MATRIX (ETİKETLİ)
# =====================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=test_gen.class_indices.keys(),
    yticklabels=test_gen.class_indices.keys()
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# =====================================================
# 10) ROC – AUC
# =====================================================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("ROC AUC:", roc_auc)
