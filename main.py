import torch
import torchvision
from PIL import Image
import os
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader

def showRandomImages():
    random_int = random.randint(0, len(train_array))
    low_image, high_image = train_array[random_int]
    low_image = low_image.permute(1, 2, 0)
    high_image = high_image.permute(1, 2, 0)
    plt.figure(figsize=(5, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(low_image)
    plt.subplot(1, 2, 2)
    plt.imshow(high_image)
    plt.show()

train_array = []
low_res_path = './dataset/train/low_res/'
high_res_path = './dataset/train/high_res/'
files = os.listdir(low_res_path)

preprocess_image = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomGrayscale(0.3),
    transforms.RandomAdjustSharpness(0.3),
    transforms.ToTensor()
])

high_process_image = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()
])

for index, pic in enumerate(files):
    low_image_path = low_res_path + pic
    high_image_path = high_res_path + pic
    low_image = Image.open(low_image_path)
    low_image = low_image.convert('RGB')
    low_image = preprocess_image(low_image)
    high_image = Image.open(high_image_path)
    high_image = high_image.convert('RGB')
    high_image = high_process_image(high_image)
    data_tupple = (low_image, high_image) #X, y
    train_array.append(data_tupple)


class HighResModel(nn.Module):
    def __init__(self):
        super(HighResModel, self).__init__()
        self.conv_bock_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_bock_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_bock_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_bock_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_bock_5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.upscale = nn.PixelShuffle(1) #input image size is 256 hence upscalling to 1024

    def forward(self, x):
        x = self.conv_bock_1(x)
        x = self.conv_bock_2(x)
        x = self.conv_bock_3(x)
        x = self.conv_bock_4(x)
        x = self.conv_bock_5(x)
        x = self.upscale(x)
        return x


batch_data = DataLoader(train_array, batch_size=8)
model = HighResModel()

# Print the model architecture and check its parameters
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.L1Loss()
optim_fn = torch.optim.Adam(params=model.parameters(), lr=0.001)
for epoch in tqdm(torch.arange(0, 10)):
    for batch, (input, label) in enumerate(batch_data):
        model.train()
        pred = model(input)
        loss = loss_fn(pred, label)
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()

        if batch % 8 == 0:
            print(f"Loss {loss}")

torch.save(model, 'model.pth')
model.eval()
random_int = random.randint(0, len(train_array))
low_image, high_image = train_array[random_int]
output = low_image.permute(1,2,0)
plt.imshow(output)
plt.show()