import pandas as pd
import torch
import time
from glob import glob
from PIL import Image
from numpy import asarray
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F



def  from_path_to_feature(path):
    image = Image.open(path)
    data = asarray(image)
    return data.reshape(-1)
    
#Imported dataset

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


#Pre-trained RESNET50 model
class ImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 9)

        weight = self.resnet50.conv1.weight.clone()
        self.resnet50.conv1 = nn.Conv2d(10, 1, kernel_size=3, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet50.conv1.weight[:, :7] = weight
            self.resnet50.conv1.weight[:, 3] = self.resnet50.conv1.weight[:, 0]



    def forward(self, X):
        return self.resnet50(X)


#Training the model
def train(model, dataloader, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch in dataloader:
            feature, label = batch
            feature = feature.unsqueeze(1)
            print(feature.shape)
            prediction = model(feature)
            loss = F.cross_entropy(prediction, label)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            batch += 1
    


#Putting everything together for the training loop
if __name__ == '__main__':
    torchdset = ResizedDataset()
    imagedsloader = DataLoader(torchdset, batch_size=10, shuffle=True)
    model = ImageClassifier()
    train(model, imagedsloader)

    torch.save(model.state_dict(), f'/Users/macbook/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/model_evaluation/{time.time}')
    sd = (model.state_dict())
    torch.save(sd['fc.weight'], f'/Users/macbook/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/model_evaluation/weights/{time.time}')


