import torch
import torch.nn as nn
import numpy as np
import joblib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from ai.preprocessing import preprocess_data
from ai.gnn import GATModel


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


X_train, X_test, y_train, y_test, scaler, le, feature_cols = preprocess_data(
    csv_path="data/Adaptive_Lab_Guardian.csv",
    oversample_train=False  
)

# =========================
# Graph Construction
# =========================
def create_edge_index(num_nodes=5):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t()

edge_index = create_edge_index()


def create_graph(sample, label):
    x = torch.tensor(sample, dtype=torch.float).view(5, 1)
    y = torch.tensor(label, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


train_graphs = [create_graph(X_train[i], y_train[i]) for i in range(len(X_train))]
test_graphs  = [create_graph(X_test[i],  y_test[i])  for i in range(len(X_test))]
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16)


# =========================
# Model Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GATModel().to(device)
num_classes = len(le.classes_)
classifier = nn.Linear(8, num_classes).to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()


# =========================
# Training Function
# =========================
def train():
    model.train()
    classifier.train()

    total_loss = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        embedding = model(data.x, data.edge_index, data.batch)
        out = classifier(embedding)

        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# Testing Function
# =========================
def test():
    model.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            embedding = model(data.x, data.edge_index, data.batch)
            out = classifier(embedding)

            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    return correct / total


# =========================
# Training Loop
# =========================
best_acc = 0

for epoch in range(20):
    loss = train()
    acc = test()

    if acc > best_acc:
        best_acc = acc

    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

print("Best Accuracy:", best_acc)


# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "ai/models/gnn.pth")


# =========================
# Extract Embeddings
# =========================
model.eval()
embeddings = []

with torch.no_grad():
    for data in train_loader:
        data = data.to(device)

        emb = model(data.x, data.edge_index, data.batch)
        embeddings.append(emb.cpu().numpy())

embeddings = np.vstack(embeddings)


# =========================
# PCA
# =========================
pca = PCA(n_components=3)
pca.fit(embeddings)

joblib.dump(pca, "ai/models/pca.pkl")


print("Training complete. GAT and PCA Models saved successfully.")