import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        self.conv1 = nn.Conv2d(3, 40, 2, 1)
        self.conv2 = nn.Conv2d(40, 40, 2, 1)
        self.conv3 = nn.Conv2d(40, 40, 2, 1)
        self.conv4 = nn.Conv2d(40, 40, 2, 1)

        self.fc1 = nn.Linear(64*48, 100)
        self.fc2 = nn.Linear(100, 10)
        
        self.fc_m2 = nn.Linear(360, 100)
        self.fc_m3 = nn.Linear(40, 100)

        self.fc_m4 = nn.Linear(100, 100)
        
        self.fc_m5_a = nn.Linear(1000, 1000)
        self.fc_m5_b = nn.Linear(1000,1000)
        self.fc_m5_c = nn.Linear(1000,10)
        
        self.dropout1 = nn.Dropout2d(0.5)
        
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
        elif mode == 6:
            self.forward = self.model_6
        elif mode == 7:
            self.forward = self.model_7
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-7")
            exit(0)

    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One hidden layer 
        
        X = torch.flatten(X, start_dim=1)
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.fc2(X))
                
        return X

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        
        X = torch.sigmoid(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = torch.sigmoid(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = torch.sigmoid(self.fc_m2(X))
        X = torch.sigmoid(self.fc2(X))
        
        return X

    # Replace sigmoid with relu.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with relu.
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m2(X))
        X = F.relu(self.fc2(X))
        
        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with relu.
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m2(X))
        X = F.relu(self.fc_m4(X))
        X = F.relu(self.fc2(X))
        X = F.relu(X)
        
        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with relu.
        # and  + Dropout.
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m5_a(X))
        X = self.dropout1(X)
        X = F.relu(self.fc_m5_b(X))
        X = self.dropout1(X)
        X = F.relu(self.fc_m5_c(X))
        
        return X

    def model_6(self, X):
        # ======================================================================
        # Three convolutional layers + two fully connected layers, with relu.

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m2(X))
        X = F.relu(self.fc_m4(X))
        X = F.relu(self.fc2(X))
        X = F.relu(X)

        return X

    def model_7(self, X):
        # ======================================================================
        # Four convolutional layers + two fully connected layers, with relu.

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m3(X))
        X = F.relu(self.fc_m4(X))
        X = F.relu(self.fc2(X))
        X = F.relu(X)

        return X
