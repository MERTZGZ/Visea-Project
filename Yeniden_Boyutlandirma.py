import cv2
import os

# Resimlerin bulunduğu dizin
input_path = 'C:/Users/Mert/Desktop/VISEA/Train/Images'
# Dönüştürülmüş resimlerin kaydedileceği dizin
output_path = 'C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images'

# Resim boyutları
width = 1224
height = 370

# Tüm resimleri dönüştür
for filename in os.listdir(input_path):
    # Resmi oku
    img = cv2.imread(os.path.join(input_path, filename))
    # Resmi boyutlarını değiştir
    img = cv2.resize(img, (width, height))
    # Dönüştürülmüş resmi kaydet
    cv2.imwrite(os.path.join(output_path, filename), img)