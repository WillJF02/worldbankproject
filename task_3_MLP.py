import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

#mlp model initialisation
class MLP(nn.Module):
    def __init__(self, input_size, hidden_Sizes, output_Size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_Sizes[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_Sizes[0], hidden_Sizes[1])
        self.fc3 = nn.Linear(hidden_Sizes[1], hidden_Sizes[2])
        self.fc4 = nn.Linear(hidden_Sizes[2],hidden_Sizes[3])
        self.fc5 = nn.Linear(hidden_Sizes[3], output_Size)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

def train_MLP(X_Train, y_Train, kf, learning_rates, input_Size, hidden_Sizes, output_Size):
    best_lr = None
    best_Accuracy = 0
    for lr in learning_rates:
        accuracy = []
        for train_Index, val_Index in kf.split(X_Train):
            X_train_fold, X_val_fold = X_Train.iloc[train_Index], X_Train.iloc[val_Index]
            Y_train_fold, Y_val_fold = y_Train.iloc[train_Index], y_Train.iloc[val_Index]

            #model training
            model = MLP(input_Size, hidden_Sizes, output_Size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            X_train_tensor = torch.tensor(X_train_fold.values, dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_train_fold.values, dtype=torch.long).squeeze()
            X_val_tensor = torch.tensor(X_val_fold.values, dtype=torch.float32)
            Y_val_tensor = torch.tensor(Y_val_fold.values, dtype=torch.long).squeeze()

            #training Loop
            iter = 500
            for i in range(iter):
                outputs = model(X_train_tensor)
                loss = criterion(outputs, Y_train_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i) % 50 == 0:
                    print(f"iter [{i}/{iter}], Loss: {loss.item():.4f}")
            
            #evaluation on y train set
            with torch.no_grad():
                outputs_val = model(X_val_tensor)
                predictions = torch.argmax(outputs_val, dim=1).numpy()
                accuracy.append(accuracy_score(Y_val_tensor, predictions))

        mean_Accuracy = np.mean(accuracy)

        if mean_Accuracy > best_Accuracy:
            best_Accuracy = mean_Accuracy
            best_lr = lr
            best_model = model
            

    print(f"Best Learning Rate for MLP: {best_lr}, Accuracy: {best_Accuracy}")
    return best_lr

#function to use the best lr on the mlp
def reTrain_MLP(X_Train, y_Train, best_lr, input_Size, hidden_Sizes, output_Size):
    trained_Model = MLP(input_Size, hidden_Sizes, output_Size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trained_Model.parameters(), lr=best_lr)

    X_train_tensor = torch.tensor(X_Train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_Train.values, dtype=torch.long).squeeze()

    iter = 500
    for i in range(iter):
        outputs = trained_Model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return trained_Model

#function to test mlp
def test_MLP(X_Test, y_Test, best_Model):
    #get the confusion matrix for the MLP
    X_test_tensor = torch.tensor(X_Test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_Test.values, dtype=torch.long).squeeze()

    with torch.no_grad():
        outputs_test = best_Model(X_test_tensor)
        predictions_test = torch.argmax(outputs_test, dim=1).numpy()
        accuracy = accuracy_score(y_test_tensor, predictions_test)
        print(f"MLP test accuracy: {accuracy}")
    conf_Matrix = confusion_matrix(y_test_tensor, predictions_test)

    sns.heatmap(conf_Matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1', '2', '3'],
            yticklabels=['0', '1', '2', '3'])
    plt.title("Confusion matrix for MLP")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()