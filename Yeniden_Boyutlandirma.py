"""
Bu Python kodu, OpenCV kütüphanesi kullanılarak resim boyutlarını değiştirerek resimlerin yeniden boyutlandırılmasını sağlar. 
Kod, belirtilen bir dizindeki tüm resimlerin boyutunu değiştirir ve yeni boyutlandırılmış resimleri başka bir dizine kaydeder.

Kodun açıklaması şöyledir:

import cv2: OpenCV kütüphanesi yüklenir.
import os: Bu, işletim sistemi ile ilgili fonksiyonları kullanabilmemizi sağlar.
input_path: Resimlerin bulunduğu dizin belirtilir.
output_path: Yeni boyutlandırılmış resimlerin kaydedileceği dizin belirtilir.
width: Yeni resim genişliği belirtilir.
height: Yeni resim yüksekliği belirtilir.
for filename in os.listdir(input_path): Döngü, belirtilen dizindeki tüm dosyaları okumak için kullanılır.
img = cv2.imread(os.path.join(input_path, filename)): cv2.imread() fonksiyonu, belirtilen yolu kullanarak resim dosyasını okur.
img = cv2.resize(img, (width, height)): cv2.resize() fonksiyonu, resim boyutunu değiştirir.
cv2.imwrite(os.path.join(output_path, filename), img): cv2.imwrite() fonksiyonu, yeni boyutlandırılmış resmi belirtilen yere kaydeder.

"""
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
