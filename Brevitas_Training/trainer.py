import os

import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support


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

        train_loss = 0

        true_labels = []
        predictions = []
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
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            true_labels.extend(list(y.cpu().detach().numpy()))
            predictions.extend(list(pred.argmax(1).cpu().detach().numpy()))

        accuracy = accuracy_score(y_true=true_labels, y_pred=predictions)
        # ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=true_labels,
                                                                   y_pred=predictions,
                                                                   average='macro')
        train_loss /= num_batches
        return accuracy, f1, train_loss

    def validate_one_epoch(self):
        num_batches = len(self.val_dataloader)
        self.model.eval()
        val_loss = 0

        true_labels = []
        predictions = []
        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()

                true_labels.extend(list(y.cpu().detach().numpy()))
                predictions.extend(list(pred.argmax(1).cpu().detach().numpy()))

        accuracy = accuracy_score(y_true=true_labels, y_pred=predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=true_labels,
                                                                   y_pred=predictions,
                                                                   average='macro')
        val_loss /= num_batches

        print(f"Validation Error: \n Accuracy (Top1): {accuracy:>0.2f}%, F1 (macro): {f1:>0.2f}%, Avg loss: {val_loss:>8f}\n")
        return accuracy, f1, val_loss

    def test(self, report_name):
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss = 0

        true_labels = []
        predictions = []
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

                true_labels.extend(list(y.cpu().detach().numpy()))
                predictions.extend(list(pred.argmax(1).cpu().detach().numpy()))

        accuracy = accuracy_score(y_true=true_labels, y_pred=predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=true_labels,
                                                                   y_pred=predictions,
                                                                   average='macro')
        test_loss /= num_batches

        print(f"Test Error:\nAccuracy (Top1): {accuracy:>0.2f}%, F1 (macro): {f1:>0.2f}, Avg loss: {test_loss:>8f}\n")

        # CREATE CLASSIFICATION REPORT AND EXPORT IT
        # ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        report = classification_report(y_true=true_labels, y_pred=predictions)
        clf_report_folder_path = "Classification Reports"
        if not os.path.exists(clf_report_folder_path):  # Check if the folder exists, if not, create it
            os.makedirs(clf_report_folder_path)

        with open(os.path.join(clf_report_folder_path, report_name),  "w") as text_file:
            text_file.write(report)

        return accuracy, precision, recall, f1, test_loss


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
