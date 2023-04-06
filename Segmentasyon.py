'''

import cv2
import numpy as np

# G�r�nt�y� oku
image = cv2.imread("C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images/train_0.png")

# G�r�nt�y� HSV renk uzay�na d�n��t�r
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Ara� ve insan renk aral���n� belirle
lower_vehicle_color = np.array([0, 0, 0], dtype=np.uint8)  # �rnek bir ara� rengi
upper_vehicle_color = np.array([180, 255, 50], dtype=np.uint8)  # �rnek bir ara� rengi
lower_human_color = np.array([0, 20, 70], dtype=np.uint8)  # �rnek bir insan rengi
upper_human_color = np.array([20, 255, 255], dtype=np.uint8)  # �rnek bir insan rengi

# Renk aral���na g�re bir maske olu�tur
vehicle_mask = cv2.inRange(hsv_image, lower_vehicle_color, upper_vehicle_color)
human_mask = cv2.inRange(hsv_image, lower_human_color, upper_human_color)

# Maskeyi g�r�nt�ye uygula
segmented_vehicle = cv2.bitwise_and(image, image, mask=vehicle_mask)
segmented_human = cv2.bitwise_and(image, image, mask=human_mask)

# Sonu�lar� g�ster
cv2.imshow("Orjinal G�r�nt�", image)
cv2.imshow("Ara� Segmentasyonu", segmented_vehicle)
cv2.imshow("�nsan Segmentasyonu", segmented_human)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


'''

import cv2
import numpy as np

# G�r�nt�y� oku
image = cv2.imread("C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images/train_0.png")

# G�r�nt�y� HSV renk uzay�na d�n��t�r
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Ara� ve insan renk aral���n� belirle
lower_vehicle_color = np.array([0, 0, 0], dtype=np.uint8)  # �rnek bir ara� rengi
upper_vehicle_color = np.array([180, 255, 50], dtype=np.uint8)  # �rnek bir ara� rengi
lower_human_color = np.array([0, 20, 70], dtype=np.uint8)  # �rnek bir insan rengi
upper_human_color = np.array([20, 255, 255], dtype=np.uint8)  # �rnek bir insan rengi

# Renk aral���na g�re bir maske olu�tur
vehicle_mask = cv2.inRange(hsv_image, lower_vehicle_color, upper_vehicle_color)
human_mask = cv2.inRange(hsv_image, lower_human_color, upper_human_color)

# Maskeyi g�r�nt�ye uygula
segmented_vehicle = cv2.bitwise_and(image, image, mask=vehicle_mask)
segmented_human = cv2.bitwise_and(image, image, mask=human_mask)

# Sonu�lar� g�ster
cv2.imshow("Orjinal G�r�nt�", image)
cv2.imshow("Ara� Segmentasyonu", segmented_vehicle)
cv2.imshow("�nsan Segmentasyonu", segmented_human)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

# G�r�nt�y� oku
image = cv2.imread("C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images/train_0.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# G�r�nt�y� tens�re d�n��t�r
transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image)
image_tensor = torch.unsqueeze(image_tensor, 0)  # Batch boyutunu ekleyerek tens�r� 4D hale getir

# Evri�imli sinir a�� (CNN) modelini y�kle
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# G�r�nt�y� modelde ileri ge�ir ve tahminleri al
with torch.no_grad():
    output = model(image_tensor)['out']
    output = torch.argmax(output, dim=1)  # S�n�f indekslerini al

# S�n�f indekslerini numpy dizisine d�n��t�r
segmented_image = output.numpy()[0]

# Sonu�lar� g�ster
cv2.imshow("Orjinal G�r�nt�", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow("Segmentasyon Sonucu", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

