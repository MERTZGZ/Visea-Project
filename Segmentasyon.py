'''

import cv2
import numpy as np

# Görüntüyü oku
image = cv2.imread("C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images/train_0.png")

# Görüntüyü HSV renk uzayýna dönüþtür
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Araç ve insan renk aralýðýný belirle
lower_vehicle_color = np.array([0, 0, 0], dtype=np.uint8)  # Örnek bir araç rengi
upper_vehicle_color = np.array([180, 255, 50], dtype=np.uint8)  # Örnek bir araç rengi
lower_human_color = np.array([0, 20, 70], dtype=np.uint8)  # Örnek bir insan rengi
upper_human_color = np.array([20, 255, 255], dtype=np.uint8)  # Örnek bir insan rengi

# Renk aralýðýna göre bir maske oluþtur
vehicle_mask = cv2.inRange(hsv_image, lower_vehicle_color, upper_vehicle_color)
human_mask = cv2.inRange(hsv_image, lower_human_color, upper_human_color)

# Maskeyi görüntüye uygula
segmented_vehicle = cv2.bitwise_and(image, image, mask=vehicle_mask)
segmented_human = cv2.bitwise_and(image, image, mask=human_mask)

# Sonuçlarý göster
cv2.imshow("Orjinal Görüntü", image)
cv2.imshow("Araç Segmentasyonu", segmented_vehicle)
cv2.imshow("Ýnsan Segmentasyonu", segmented_human)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


'''

import cv2
import numpy as np

# Görüntüyü oku
image = cv2.imread("C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images/train_0.png")

# Görüntüyü HSV renk uzayýna dönüþtür
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Araç ve insan renk aralýðýný belirle
lower_vehicle_color = np.array([0, 0, 0], dtype=np.uint8)  # Örnek bir araç rengi
upper_vehicle_color = np.array([180, 255, 50], dtype=np.uint8)  # Örnek bir araç rengi
lower_human_color = np.array([0, 20, 70], dtype=np.uint8)  # Örnek bir insan rengi
upper_human_color = np.array([20, 255, 255], dtype=np.uint8)  # Örnek bir insan rengi

# Renk aralýðýna göre bir maske oluþtur
vehicle_mask = cv2.inRange(hsv_image, lower_vehicle_color, upper_vehicle_color)
human_mask = cv2.inRange(hsv_image, lower_human_color, upper_human_color)

# Maskeyi görüntüye uygula
segmented_vehicle = cv2.bitwise_and(image, image, mask=vehicle_mask)
segmented_human = cv2.bitwise_and(image, image, mask=human_mask)

# Sonuçlarý göster
cv2.imshow("Orjinal Görüntü", image)
cv2.imshow("Araç Segmentasyonu", segmented_vehicle)
cv2.imshow("Ýnsan Segmentasyonu", segmented_human)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

# Görüntüyü oku
image = cv2.imread("C:/Users/Mert/Desktop/VISEA/Train/Yeni_Images/train_0.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü tensöre dönüþtür
transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image)
image_tensor = torch.unsqueeze(image_tensor, 0)  # Batch boyutunu ekleyerek tensörü 4D hale getir

# Evriþimli sinir aðý (CNN) modelini yükle
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Görüntüyü modelde ileri geçir ve tahminleri al
with torch.no_grad():
    output = model(image_tensor)['out']
    output = torch.argmax(output, dim=1)  # Sýnýf indekslerini al

# Sýnýf indekslerini numpy dizisine dönüþtür
segmented_image = output.numpy()[0]

# Sonuçlarý göster
cv2.imshow("Orjinal Görüntü", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow("Segmentasyon Sonucu", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

