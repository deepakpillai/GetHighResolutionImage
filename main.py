import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from torch.optim import lr_scheduler
import os
import cv2
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

pre_process = transforms.Compose([
    transforms.Resize(size=(320, 320)),
    transforms.ToTensor(),
])

resize_transforms = transforms.Compose([
    transforms.Resize(size=(336, 336)),
    transforms.ToTensor()
])

# https://www.kaggle.com/datasets/quadeer15sh/image-super-resolution-from-unsplash?resource=download
dataset = []

legend = pd.read_csv('./image_data.csv')
legent_df = pd.DataFrame(legend)
# shuffle the DataFrame rows
# legent_df = legent_df.sample(frac = 1)
for index in legent_df.index:
    low_image = legent_df['low_res'][index]
    high_res = legent_df['high_res'][index]

    low_path = "./dataset/low_res/"
    high_path = "./dataset/high_res/"

    low_res_image_path = low_path + low_image
    high_res_image_path = high_path + high_res


    low_res_image = pre_process(Image.open(low_res_image_path))
    high_res_image = resize_transforms(Image.open(high_res_image_path))
    low_res_image = low_res_image.to(device)
    high_res_image = high_res_image.to(device)
    image_tupple = (low_res_image, high_res_image)
    dataset.append(image_tupple)
    # if index == 1:
    #     break


# low, high = dataset[8]
# low = low.permute(1,2,0).cpu()
# plt.subplot(1,2,1)
# plt.imshow(low)
# plt.subplot(1,2,2)
# plt.imshow(high.permute(1,2,0).cpu())
# plt.show()

# low_res_path = './dataset_256/low_res/'
# high_res_path = './dataset_256/high_res/'
# files = os.listdir(low_res_path)

# for index, pic in enumerate(files):
#     low_image_path = low_res_path + pic
#     high_image_path = high_res_path + pic
#     low_image = Image.open(low_image_path)
#     low_image = low_image.convert('RGB')
#     low_image = pre_process(low_image)
#     high_image = Image.open(high_image_path)
#     high_image = high_image.convert('RGB')
#     high_image = resize_transforms(high_image)
#     data_tupple = (low_image, high_image) #X, y
#     dataset.append(data_tupple)


batch_size = 8
batch_data = DataLoader(dataset, batch_size=batch_size)



class HDModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=30, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=30, out_channels=29, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=29, out_channels=3, kernel_size=3, stride=1),
            # nn.PixelShuffle(2)
            
        )
    
    def forward(self, x):
        data = self.cnnblock(x)
        return data
    
# testing model
# sample_data = torch.randn(3, 256, 256)
# model = HDModel()
# model(sample_data)

total_epoch = 20
model = HDModel().to(device)
loss_fn = nn.MSELoss()
optim_fn = torch.optim.AdamW(model.parameters(), lr=0.001)
# scheduler = lr_scheduler.LinearLR(optimizer=optim_fn, start_factor=1.0, end_factor=0.5, total_iters=total_epoch)
model.train()
for epoch in tqdm(range(0, total_epoch)):
    batch_loss = 0
    for batch, (feature, label) in enumerate(dataset):
        
        feature = feature.to(device)
        label = label.to(device)
        pred = model(feature)
        loss = loss_fn(pred, label)
        batch_loss = batch_loss + loss
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()

        if batch % 8 == 0:
            print(f"batch_loss: {batch_loss/batch_size}")

    # before_lr = optim_fn.param_groups[0]['lr']
    # scheduler.step()
    # after_lr = optim_fn.param_groups[0]['lr']
    # print(f"Epoch: {epoch}, before_lr: {before_lr}, after_lr: {after_lr}")

torch.save(model, "./model.pth")
torch.save(model.state_dict(), "./model_state_dict.pth")