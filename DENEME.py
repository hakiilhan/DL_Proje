import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Tüm görüntüleri 0-1 arasına normalize ediyoruz
rescale = 1./255

# --- Veri Artırma (Train için) [cite: 39] ---
# Rotasyon, kayma, yatay çevirme gibi teknikleri kullanın
train_datagen = ImageDataGenerator(
    rescale=rescale,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Normalizasyon (Validation ve Test için) ---
# Sadece yeniden ölçeklendirme yapılır, artırma yapılmaz
val_test_datagen = ImageDataGenerator(rescale=rescale)

# Model giriş boyutu ve toplu iş boyutu (batch size)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2 # Veri setinize göre bu sayıyı güncelleyin (örneğin 4 veya 5)

# Veri yollarını kendi klasör yapınıza göre güncelleyin
train_dir = r'C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\train'
validation_dir = r'C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\val'
test_dir = r'C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Çok sınıflı sınıflandırma için
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Test sonuçlarının tutarlılığı için karıştırmayın
)

def create_cnn_model(input_shape, num_classes, l2_lambda=0.0001):
    model = Sequential([
        # 1. Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(), # Batch Normalization (BN) 
        MaxPooling2D(2, 2),

        # 2. Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # 3. Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3), # Dropout [cite: 31, 32]

        # Fully Connected (Dense) Layers
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_lambda)), # L2 Regularization [cite: 37, 42]
        BatchNormalization(),
        Dropout(0.5),

        # Output Layer (Softmax) [cite: 82]
        Dense(num_classes, activation='softmax')
    ])
    return model

# Modeli oluşturma
model = create_cnn_model(IMG_SIZE + (3,), NUM_CLASSES, l2_lambda=0.0001)

# Model özetini görselleştirin
model.summary()

# Optimizer: Adam [cite: 52]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Kayıp fonksiyonu: Çok sınıflı sınıflandırma için Categorical Crossentropy
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 1. Early Stopping (Erken Durdurma) [cite: 67, 69]
# Monitörlenen metrik: validation loss [cite: 70]
# Patience: 10 (10 epoch boyunca iyileşme olmazsa durdur) [cite: 72]
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 2. Learning Rate Scheduler (LR Zamanlayıcı) [cite: 59, 62]
# ReduceLROnPlateau kullanıyoruz: val_loss 5 epoch boyunca iyileşmezse LR'yi 0.2 ile çarp 
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks_list = [early_stopping, lr_scheduler]

EPOCHS = 50 # Erken durdurma kullanılacağı için yüksek bir değer verilebilir

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks_list
)

# Eğitim vs. Doğrulama Eğrileri 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Benzer bir kodu accuracy için de yazmalısınız.

# Modelin performansını test veri seti üzerinde değerlendirme
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Daha sonra Karmaşıklık Matrisi (Confusion Matrix), F1-Skor vb. metrikleri hesaplamak için
# predictions = model.predict(test_generator)
# predicted_classes = tf.argmax(predictions, axis=1)
# true_classes = test_generator.classes
# ... (sklearn.metrics kütüphanesi ile hesaplamalar yapılabilir)