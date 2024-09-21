import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torch import nn
import pickle


def create_model(data):
    NUM_CLASSES = 1
    NUM_FEATURES = 30
    scaler=StandardScaler()
    X=data.drop(["diagnosis"],axis=1)
    y=data["diagnosis"]
    X = scaler.fit_transform(X)  # Scale the input features
    X = torch.from_numpy(X).float()  # Now convert the scaled data to a tensor
    y = torch.from_numpy(y.values).float().unsqueeze(1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    def accuracy_fn(y_true, y_pred):
        y_true=y_true.squeeze().long()
        y_pred=y_pred.squeeze().long()
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
    
    class CancerModel(nn.Module):
        def __init__(self, input_features, output_features, hidden_units=8):
            super().__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_units),
                nn.ReLU(), 
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=output_features),
            )

        def forward(self, x):
            return self.linear_layer_stack(x)
    model_1 = CancerModel(input_features=NUM_FEATURES,output_features=NUM_CLASSES,hidden_units=8)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)
    torch.manual_seed(42)
    epochs = 100

    for epoch in range(epochs):
        y_logits = model_1(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train.squeeze())
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_1.eval()
        with torch.inference_mode():
            test_logits = model_1(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test.squeeze())
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        # if epoch % 10 == 0:
        #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
    return model_1,scaler


def get_clean_data():
    data=pd.read_csv("data/data.csv")
    data["diagnosis"]=data['diagnosis'].map({'M':1,'B':0})
    data=data.drop(['id'],axis=1)
    return data

def main():
    data_pass=get_clean_data()
    model,scaler=create_model(data_pass)
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
    PATH="model/saved_model.pt"
    torch.save(model.state_dict(),PATH)
    

if __name__=='__main__':
    main()