import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.svm import SVC


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

# Flatten dos dados 2D para 1D
X_train_flatten = X_train_tensor.view(X_train_tensor.size(0), -1)
X_val_flatten = X_val_tensor.view(X_val_tensor.size(0), -1)
X_test_flatten = X_test_tensor.view(X_test_tensor.size(0), -1)

# Treinamento e avaliação de KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_val_predictions = knn_model.predict(X_val)
knn_accuracy = accuracy_score(y_val, knn_val_predictions)
knn_accuracy_p = (knn_accuracy * 100).round(2)
print(f"KNN Accuracy on Validation Set: {knn_accuracy_p}%")

# Treinamento e avaliação de Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_val_predictions = nb_model.predict(X_val)
nb_accuracy = accuracy_score(y_val, nb_val_predictions)
nb_accuracy_p = (nb_accuracy * 100).round(2)
print(f"Naive Bayes Accuracy on Validation Set: {nb_accuracy_p}%")

# Treinamento e avaliação de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_val_predictions = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_val_predictions)
rf_accuracy_p = (rf_accuracy * 100).round(2)
print(f"Random Forest Accuracy on Validation Set: {rf_accuracy_p}%")

# Treinamento e avaliação de SVM
svm_model = SVC(kernel="linear", C=1)
svm_model.fit(X_train, y_train)
svm_val_predictions = svm_model.predict(X_val)
svm_accuracy = accuracy_score(y_val, svm_val_predictions)
svm_accuracy_p = (svm_accuracy * 100).round(2)
print(f"SVM Accuracy on Validation Set: {svm_accuracy_p}%")

# Compare as acurácias dos modelos
print("\nComparação de Acurácias:")
print(f"KNN Accuracy: {knn_accuracy_p}%")
print(f"Naive Bayes Accuracy: {nb_accuracy_p}%")
print(f"Random Forest Accuracy: {rf_accuracy_p}%")
print(f"SVM Accuracy: {svm_accuracy_p}%")
# Agora, você pode escolher o modelo com a melhor acurácia no conjunto de validação e avaliá-lo no conjunto de teste.
