import cv2
import os

# Resimlerin bulundu�u dizin
input_path = 'C:/Users/Mert/Desktop/VISEA/Train/Images'
# D�n��t�r�lm�� resimlerin kaydedilece�i dizin
output_path = 'C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images'

# Resim boyutlar�
width = 1224
height = 370

# T�m resimleri d�n��t�r
for filename in os.listdir(input_path):
    # Resmi oku
    img = cv2.imread(os.path.join(input_path, filename))
    # Resmi boyutlar�n� de�i�tir
    img = cv2.resize(img, (width, height))
    # D�n��t�r�lm�� resmi kaydet
    cv2.imwrite(os.path.join(output_path, filename), img)