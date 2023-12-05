import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import numpy as np


# Carregar dados a partir de um arquivo CSV
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    features = data.drop("final_result", axis=1).values
    labels = data["final_result"].values
    return features, labels


# Especificar o caminho para o arquivo CSV
csv_path = "dataframe_filna.csv"  # Substitua pelo caminho correto do seu arquivo CSV

# Carregar dados
X, y = load_data(csv_path)

# Adicionar uma coluna de zeros
X = np.hstack((X, np.zeros((X.shape[0], 1))))

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir o conjunto de dados em treinamento, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Remodelar os dados para uma matriz 2D
num_features = 164
X_train_2d = X_train.reshape(-1, num_features // 4, 4)
X_val_2d = X_val.reshape(-1, num_features // 4, 4)
X_test_2d = X_test.reshape(-1, num_features // 4, 4)

# Converter para tensores do PyTorch
X_train_tensor = torch.FloatTensor(X_train_2d)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_2d)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_2d)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Definir transformações de aumento de dados
data_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandAugment(),
        transforms.RandomGrayscale(),
        transforms.RandomInvert(),
        # Adicione outras transformações conforme necessário
    ]
)

# Criar conjuntos de dados usando TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Aplicar aumentação de dados apenas no conjunto de treinamento
train_dataset.transform = data_transform

# Criar DataLoader para treinamento, validação e teste
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# Definir a arquitetura da CNN
class CNN(nn.Module):
    def __init__(self, input_channels, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=64,  # Ajuste conforme necessário com base no número de canais de saída da camada anterior
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm1d(128)

        conv_output_size = self._get_conv_output_size(
            input_channels, num_features // 4, 4
        )

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(128, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output_size(self, input_channels, height, width):
        x = torch.randn(1, input_channels, height * width)
        x = self.conv1(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


# Configuração do dispositivo (CPU ou GPU, se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criar instância do modelo
input_channels = 1  # Agora temos 1 canal após adicionar a coluna de zeros
model = CNN(input_channels).to(device)

# Definir função de perda e otimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Treinamento
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
    ):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_dataset)
    print(f"Training Loss after Epoch {epoch + 1}: {avg_loss}, Accuracy: {accuracy}")


# Avaliação no conjunto de teste
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calcular a acurácia
correct = sum(p == l for p, l in zip(all_predictions, all_labels))
accuracy = (correct / len(all_labels)) * 100
print(f"Test accuracy: {accuracy}%")
