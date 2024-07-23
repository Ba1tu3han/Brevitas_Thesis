import torch


class Trainer:

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 device,
                 train_dataloader,
                 val_dataloader=None,
                 test_dataloader=None,
                 sample_size=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.sample_size = sample_size

    def train_one_epoch(self):  # training function
        if self.sample_size:
            size = min(self.sample_size, len(self.train_dataloader.dataset))
        else:
            size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        self.model.train()

        train_loss, correct = 0, 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()


            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        return accuracy, train_loss

    def validate_one_epoch(self):
        if self.sample_size:
            size = min(self.sample_size, len(self.val_dataloader.dataset))
        else:
            size = len(self.val_dataloader.dataset)

        num_batches = len(self.val_dataloader)
        self.model.eval()
        val_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        print(f"Test Error: \n Accuracy (Top1): {accuracy:>0.2f}%, Avg loss: {val_loss:>8f}\n")
        return accuracy, val_loss

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        print(f"Test Error: \n Accuracy (Top1): {accuracy:>0.2f}%, Avg loss: {test_loss:>8f}\n")
        return accuracy, test_loss


class EarlyStopper:
    # ref: https://stackoverflow.com/a/73704579/15169035
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
