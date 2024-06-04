import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data.drop('y', axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Применение преобразований к тренировочным и тестовым данным
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if sparse.issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()
if sparse.issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_train_processed, y_train)
test_dataset = MyDataset(X_test_processed, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        return x

class MediumModel(nn.Module):
    def __init__(self, input_dim):
        super(MediumModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

class ComplexModel(nn.Module):
    def __init__(self, input_dim):
        super(ComplexModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

models = [SimpleModel(X_train_processed.shape[1]), MediumModel(X_train_processed.shape[1]), ComplexModel(X_train_processed.shape[1])]

criterion = nn.BCELoss()
optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in models]

train_losses = {0: [], 1: [], 2: []}
test_losses = {0: [], 1: [], 2: []}
train_accuracies = {0: [], 1: [], 2: []}
test_accuracies = {0: [], 1: [], 2: []}

num_epochs = 10
for epoch in range(num_epochs):
    for i, model in enumerate(models):
        optimizer = optimizers[i]
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(torch.tensor(X_train_processed, dtype=torch.float32))
            train_acc = ((train_preds.squeeze() > 0.5) == torch.tensor(y_train.values, dtype=torch.float32)).float().mean()
            test_preds = model(torch.tensor(X_test_processed, dtype=torch.float32))
            test_acc = ((test_preds.squeeze() > 0.5) == torch.tensor(y_test.values, dtype=torch.float32)).float().mean()
            train_accuracies[i].append(train_acc.item())
            test_accuracies[i].append(test_acc.item())

        print(f'Model {i+1}, Epoch {epoch+1}/{num_epochs}, Train accuracy: {train_accuracies[i][-1]}, Test accuracy: {test_accuracies[i][-1]}')

plt.figure(figsize=(12, 6))

for i, model in enumerate(models):
    plt.plot(range(1, num_epochs + 1), train_accuracies[i], label=f'Model {i+1} Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies[i], label=f'Model {i+1} Test Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()