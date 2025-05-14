import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


img1 = Image.open('/Users/kiarabauerle/Documents/Hanyang/AI study group/HAI_Assignment_KiaraB/lecture05/image1.jpg')   
img2 = Image.open('/Users/kiarabauerle/Documents/Hanyang/AI study group/HAI_Assignment_KiaraB/lecture05/image2.jpg')  


transform = transforms.ToTensor()

img1_tensor1 = transform(img1)
img2_tensor2 = transform(img2)

print(img1_tensor1.shape)  
print(img1_tensor1.dtype)  
print(img2_tensor2.shape)  
print(img2_tensor2.dtype)  

plt.imshow(img1_tensor1.permute(1, 2, 0))  
plt.show()
plt.imshow(img2_tensor2.permute(1, 2, 0))
plt.show()

res1 = img1_tensor1 * img2_tensor2

plt.imshow(res1.permute(1, 2, 0))  
plt.show()

res2 = torch.matmul(img1_tensor1.view(3, -1).T, img2_tensor2.view(3, -1))

print(f"The matrix multiplication of image1 and image2 is: \n{res2}")