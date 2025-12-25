import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 1. VERİ HAZIRLAMA VE ÖN İŞLEME (Bölüm 4) [cite: 98, 107]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Dosya yollarını 'r' (raw string) kullanarak güncelleyin
train_dir = r'C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\train'
validation_dir = r'C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\val'
test_dir = r'C:\Users\hknil\OneDrive\Masaüstü\JUPYTER\chest_xray_balanced\test'

# Veri Artırma (Data Augmentation) - Overfitting'i önlemek için [cite: 39, 159]
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# 2. MODEL MİMARİSİ (Bölüm 2.1) [cite: 14, 21, 24]
# L2 regularizer eklenerek ağırlıkların aşırı büyümesi engellenir [cite: 37]
def build_optimized_model():
    model = Sequential([
        # Blok 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(), # Eğitim stabilitesi için [cite: 28]
        MaxPooling2D(2, 2),
        
        # Blok 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Blok 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3), # Overfitting önlemi [cite: 31]

        # Tam Bağlantılı Katmanlar
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)), # L2 Regularization [cite: 37]
        BatchNormalization(),
        Dropout(0.5), # Daha güçlü dropout [cite: 32]
        
        # Çıkış Katmanı
        Dense(train_generator.num_classes, activation='softmax') # [cite: 82]
    ])
    return model

model = build_optimized_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. EĞİTİM STRATEJİSİ VE CALLBACKS (Bölüm 2.3) [cite: 49, 58, 67]
# Underfitting'i önlemek ve hızlı sonuç almak için güncellendi
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5,              # 5 epoch iyileşme olmazsa dur 
    restore_best_weights=True, 
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,              # LR'yi yarıya indir 
    patience=2,              # Daha hassas tepki (2 epoch)
    min_lr=1e-7, 
    verbose=1
)

# 4. MODEL EĞİTİMİ 
history = model.fit(
    train_generator,
    epochs=15,               # Optimize edilmiş epoch sayısı 
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# 5. GÖRSELLEŞTİRME VE ANALİZ (Bölüm 5) [cite: 46, 112]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Analysis')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Analysis')
plt.legend()
plt.show()

# 6. TEST VE RAPORLAMA METRİKLERİ [cite: 114, 120]
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Confusion Matrix için tahminler
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('\nConfusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))