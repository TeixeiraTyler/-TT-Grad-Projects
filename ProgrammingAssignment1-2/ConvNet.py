import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        self.fcl = nn.Linear(784, 100)
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.conv2 = nn.Conv2d(40, 6, 5)
        self.fc1 = nn.Linear(6*4*4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.conv10 = nn.Conv2d(1, 40, 5)
        self.conv20 = nn.Conv2d(40, 62, 5)
        self.fc10 = nn.Linear(62*4*4, 1000)
        self.fc20 = nn.Linear(1000, 10)
        self.dropout = nn.Dropout(0.5)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.

        X = X.view(-1, self.num_flat_features(X))
        X = torch.sigmoid(self.fcl(X))

        return X

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer.

        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), 2)
        X = X.view(-1, self.num_flat_features(X))
        X = torch.sigmoid(self.fc1(X))

        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.

        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), 2)
        X = X.view(-1, self.num_flat_features(X))
        X = F.relu(self.fc1(X))

        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.

        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), 2)
        X = X.view(-1, self.num_flat_features(X))
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))

        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.

        X = F.max_pool2d(F.relu(self.conv10(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv20(X)), 2)
        X = X.view(-1, self.num_flat_features(X))
        X = F.relu(self.fc10(X))

        X = self.dropout(self.fc20(X))

        return X
