import cv2
import os

# Resimlerin bulunduðu dizin
input_path = 'C:/Users/Mert/Desktop/VISEA/Train/Images'
# Dönüþtürülmüþ resimlerin kaydedileceði dizin
output_path = 'C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images'

# Resim boyutlarý
width = 1224
height = 370

# Tüm resimleri dönüþtür
for filename in os.listdir(input_path):
    # Resmi oku
    img = cv2.imread(os.path.join(input_path, filename))
    # Resmi boyutlarýný deðiþtir
    img = cv2.resize(img, (width, height))
    # Dönüþtürülmüþ resmi kaydet
    cv2.imwrite(os.path.join(output_path, filename), img)