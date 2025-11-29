import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from climb_data import KilterDataset, climb_collate_fn
from climb_util import climb_one_hot


class NeuralNet(nn.Module):
    def __init__(self, root):
        """
        A basic MLP model for predicting grades for Kilterboard climbs
        :param root: the directory containing the Kilterboard database
        """
        super().__init__()
        self.root = root
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1403, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = climb_one_hot(x, self.root).float()
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for num, batch in enumerate(dataloader):
        grades = batch['grades'].to(device).float().unsqueeze(1)  # unsqueeze to ensure this is dim [batch_size,1]
        seqs = batch['seqs'].to(device)

        # Compute prediction error
        pred = model(seqs)
        loss = loss_fn(pred, grades)

        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if num % 100 == 0:
            loss, current = loss.item(), (num + 1) * len(seqs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, mse = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            grades = batch['grades'].to(device).float().unsqueeze(1)  # unsqueeze to ensure this is dim [batch_size,1]
            seqs = batch['seqs'].to(device)
            preds = model(seqs)
            loss = loss_fn(preds, grades).item()
            avg_loss += loss
            mse += torch.mean((preds - grades) ** 2).item()
    avg_loss /= num_batches
    mse /= size
    print(f"Test Error: \n MSE: {mse}, Avg loss: {avg_loss:>8f} \n")
    return mse, avg_loss


def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # get the full dataset and split it into train and test
    full_dataset = KilterDataset(root='data', download=True)
    training_data, test_data = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    # set up the dataloaders
    batch_size = 64
    training_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=climb_collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=climb_collate_fn)

    model = NeuralNet(root='data').to(device)

    accuracies = []
    epochs = 20
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(training_dataloader, model, loss_fn, optimizer)
        accuracy, avg_loss = test(test_dataloader, model, loss_fn)
        accuracies.append(accuracy)

    # save the weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "models/mlp_grader.pt")

    plt.plot(accuracies, label='MSE')

    plt.title('Mean Square Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()