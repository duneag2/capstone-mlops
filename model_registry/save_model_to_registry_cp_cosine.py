# save_model_to_registry.py
import os
from argparse import ArgumentParser

import mlflow
import numpy as np
import pandas as pd
import random
import psycopg2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 0. set mlflow environments
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. get data
db_connect = psycopg2.connect(
    user="myuser",
    password="mypassword",
    host="localhost",
    port=5432,
    database="mydatabase",
)

def l1_norm(a, b):
    add_inv_b = torch.mul(b, -1)
    summation = torch.add(a, add_inv_b)
    abs_val = torch.abs(summation)
    return torch.sum(abs_val).item()

def l2_norm(a, b):
    add_inv_b = torch.mul(b, -1)
    summation = torch.add(a, add_inv_b)
    square = torch.mul(summation, summation)
    sqrt = torch.sqrt(square)
    return torch.sum(sqrt).item()

def cosine_similarity(a, b):
    cos = torch.nn.functional.cosine_similarity(a, b)
    return torch.mean(cos).item()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_and_preprocess_image(image_path):
    text_str = str(image_path.split('/')[3])
    image_path = os.path.join(image_path)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text_str, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    return outputs.image_embeds.detach().cpu().numpy()

# Monday
df = pd.read_sql("SELECT * FROM cargo WHERE id <= 983", db_connect)

X = np.array(df["image_path"])
y = np.array(df["target"])

for i in range(len(X)):
    X[i] = '../api_serving/monday/' + X[i]

# print(len(X))

X_temp, X_test, y_temp, y_test = train_test_split(X, y, train_size=0.8, random_state=2022)

X_test = list(X_test)
y_test = list(y_test)

X = np.array(X_temp)
y = np.array(y_temp)

img_0 = []
img_1 = []
img_2 = []
img_3 = []
img_4 = []

for i in range(len(X)):
    if y[i] == 0:
        img_0.append(X[i])
    elif y[i] == 1:
        img_1.append(X[i])
    elif y[i] == 2:
        img_2.append(X[i])
    elif y[i] == 3:
        img_3.append(X[i])
    elif y[i] == 4:
        img_4.append(X[i])

# random.seed(256)


# rand_list = random.choices(range(0,5), k=196)

selected_pair_file_list = []
selected_pair_index_list = []

for i in range(110):
    
    if len(img_0) == 0:
        anchor_num = random.randint(1,4)
        num_list = [1, 2, 3, 4]
        num_list.remove(anchor_num)
    else:
        anchor_num = random.randint(0,4)
        num_list = [0, 1, 2, 3, 4]
        num_list.remove(anchor_num)
        sample_0 = random.sample(img_0, 1)[0]
        img_0.remove(sample_0)
    
    sample_1 = random.sample(img_1, 1)[0]
    img_1.remove(sample_1)
    sample_2 = random.sample(img_2, 1)[0]
    img_2.remove(sample_2)
    sample_3 = random.sample(img_3, 1)[0]
    img_3.remove(sample_3)
    sample_4 = random.sample(img_4, 1)[0]
    img_4.remove(sample_4)

    random_list = [sample_0, sample_1, sample_2, sample_3, sample_4]

    dist = []
    dist_num = []

    for j in num_list:
        dist_value = cosine_similarity(torch.tensor(load_and_preprocess_image(random_list[j])), torch.tensor(load_and_preprocess_image(random_list[anchor_num])))
        dist.append(dist_value)
        dist_num.append(j)
    
    dist = np.array(dist)
    closest_index = dist_num[np.argmax(dist)]

    closest_path = random_list[closest_index]
    anchor_path = random_list[anchor_num]

    selected_pair_file_list.append(closest_path)
    selected_pair_file_list.append(anchor_path)

    selected_pair_index_list.append(closest_index)
    selected_pair_index_list.append(anchor_num)

X_train = list(selected_pair_file_list)
y_train = list(selected_pair_index_list)


# Tuesday
df2 = pd.read_sql("SELECT * FROM cargo WHERE id > 983", db_connect)

X2 = np.array(df2["image_path"])
y2 = np.array(df2["target"])

for i in range(len(X2)):
    X2[i] = '../api_serving/tuesday/' + X2[i]

# print(len(X2))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.8, random_state=2022)

X_train2 = list(X_train2)
y_train2 = list(y_train2)
X_test2 = list(X_test2)
y_test2 = list(y_test2)

X_train = X_train + X_train2
y_train = y_train + y_train2

X_test = X_test + X_test2
y_test = y_test + y_test2

BATCH_SIZE = 64
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CustomDataset(file_paths=X_train, labels=y_train, transform=data_transform)
test_dataset = CustomDataset(file_paths=X_test, labels=y_test, transform=data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 2. model development and train
model = models.mobilenet_v2(pretrained=True)

for parameter in model.parameters():
    parameter.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(num_features, 5)

model_pipeline = model.to(DEVICE)

optimizer = torch.optim.Adam(model_pipeline.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : 0.95 ** epoch)

def train(model, train_loader, optimizer, log_interval):
    model.train()
    correct = 0
    pred = []

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image).squeeze(dim=1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        prediction = output.max(1, keepdim = True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()
        pred.append(prediction)

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
    train_accuracy = 100. * correct / len(train_loader.dataset)
    scheduler.step()
    return train_accuracy, pred

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE, dtype=torch.float)
            label = label.to(DEVICE)
            output = model(image).squeeze(dim=1)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, EPOCHS + 1):
    train_accuracy, train_pred_tensor = train(model, train_dataloader, optimizer, log_interval = 5)
    valid_loss, valid_accuracy = evaluate(model, test_dataloader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, valid_loss, valid_accuracy))

train_pred = []
for i in range(len(train_pred_tensor)):
  train_pred.append(np.transpose(np.array(train_pred_tensor[i]))[0])

train_pred = np.concatenate(train_pred)

print("Train Accuracy :", train_accuracy)
print("Test Accuracy :", valid_accuracy)

# 3. save model
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
args = parser.parse_args()

mlflow.set_experiment("new-exp")

signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train[:10]

with mlflow.start_run():
    mlflow.log_metrics({"train_acc": train_accuracy, "test_acc": valid_accuracy})
    mlflow.pytorch.log_model(
        pytorch_model=model, 
        artifact_path=args.model_name, 
        signature=signature, 
        input_example=input_sample)

# 4. save data
df.to_csv("data.csv", index=False)