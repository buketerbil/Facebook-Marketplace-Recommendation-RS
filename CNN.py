import pandas as pd
import torch
from glob import glob
import cv2
from PIL import Image
from numpy import asarray
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F




def  from_path_to_feature(path):
    image = Image.open(path)
    data = asarray(image)
    return data.reshape(-1)
    


class ResizedDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = self.get_data()
        self.label_dict = {}

    def get_data(self):

        dframe = pd.read_csv("/Users/macbook/Downloads/Products.csv", index_col=0).dropna().rename(columns={'id': 'product_id'})
        images_df = pd.read_csv("/Users/macbook/Downloads/Images.csv", index_col=0)

        merged_df = dframe.merge(images_df, on='product_id')

        image_data_df = pd.DataFrame({'path': glob('/Users/macbook/Downloads/resized_images/*.jpg')})

        image_data_df['id'] = image_data_df.path.str.split('/').str[-1].str.replace('.jpg', '', regex=False)

        image_data_df = image_data_df.merge(merged_df[['category', 'id']], on='id')

        image_data_df['category'] = image_data_df.category.str.split('/').str[0]

        return image_data_df

    def __getitem__(self, index):
        img = self.df.iloc[index]
        feature = from_path_to_feature(img.path)

        if img.category not in self.label_dict:
            self.label_dict[img.category] = len(self.label_dict) + 1

        label = self.label_dict[img.category]

        return (torch.tensor(feature), label)
        
    def __len__(self):
        return len(self.df)




torchdset = ResizedDataset()
imagedsloader = DataLoader(torchdset, batch_size=10, shuffle=True)

'''
for batch in imagedsloader:
    print(batch)
    break   

print(torchdset.label_dict)

'''

example = next(iter(imagedsloader))
torchdset.feature, torchdset.label = example

def train(model, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in imagedsloader:
            feature, label = batch
            prediction = model(feature)
            loss = F.cross_entropy(prediction, label)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('Loss', loss.item(), batch_idx)
            batch += 1


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 10, 5),
            torch.nn.Flatten(),
            torch.nn.Linear(10 * 5 * 5, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 9),
            torch.nn.Softmax()
        )

    def forward(self, X):
        return self.layers(X)


if __name__ == '__main__':
    torchdset = ResizedDataset()
    imagedsloader = DataLoader(torchdset, batch_size=10, shuffle=True)
    model = CNN()
    train(model)



