from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

image_path1 = r"hymenoptera_data\train\ants\0013035.jpg"
image_path2 = r"hymenoptera_data\train\bees\16838648_415acd9e3f.jpg"
img1 = Image.open(image_path1)
img2 = Image.open(image_path2)

#print(type(img))
img_array1 = np.array(img1)
img_array2 = np.array(img2)

#print(type(img_array))
#print(img_array.shape)
writer.add_image("test", img_array1, 1, dataformats='HWC')
writer.add_image("test", img_array2, 2, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y = 2x", 2 * i, i)

writer.close()