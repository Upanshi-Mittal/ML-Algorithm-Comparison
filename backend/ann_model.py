import torch
import torch.nn as nn
import torch.optim as optim


class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_ann(X_train, y_train, X_test, y_test):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    model = ANNModel(input_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(50):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_test).float().mean().item()

    return round(accuracy, 4)