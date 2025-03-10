from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

img_path = r"hymenoptera_data\train\ants\0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

#ToTensor
trans_totensor = transforms.ToTensor()
tensor_img = trans_totensor(img)
#print(type(tensor_img))
writer.add_image("tensor_img", tensor_img)

#Normalize
#print(tensor_img)
#tens_norm = transforms.Normalize([0.5,0.5,0.5], [0.5,10000000000,100000000000000])
tens_norm = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) #可以尝试注释掉这行代码，使用上面的代码，把绿色和蓝色全部搞没！
img_norm = tens_norm(tensor_img)
#print(img_norm)
writer.add_image("normal_img", img_norm)

trans_resize = transforms.Resize((52, 512))
trans_resize = transforms.Resize(60)
# If size is a sequence like(h, w), output size will be matched to this.
# If size is an int,smaller edge of the image will be matched to this number.
    #i.e, if height > width, then image will be rescaled to(size * height / width, size).
img_resize = trans_resize(tensor_img)
#print(type(img_resize))
writer.add_image("resize_img", img_resize)

writer.close()