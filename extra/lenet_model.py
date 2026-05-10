import yaml
import torch.nn as nn

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Based on Lenet Model
class Lenet(nn.Module):
    def __init__(self, n_classes=10):
        super(Lenet, self).__init__()
        # Improvement Activation
        if params['improvements']['use_advanced_activation']:
            self.activation = nn.SiLU()
        elif params['improvements']['use_relu']:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

        self.use_skip_connection = params['improvements']['use_skip_connection']

        # Layer 1: (batch_size, 3, 32, 32) -> (batch_size, 6, 16, 16)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=1, padding='same'), 
            self.activation,
            self._batch_norm(6),
            nn.MaxPool2d(2)
        )
        # Layer 2: (batch_size, 6, 16, 16) -> (batch_size, 16, 6, 6)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=1),
            self.activation,
            self._batch_norm(16),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(16*6*6, 120), 
            self.activation,
            nn.Dropout(0.2)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(120, 84), 
            self.activation,
            nn.Dropout(0.2)
        )
        self.fc_layer3 = nn.Sequential(nn.Linear(84, n_classes))

        self._initialize_weights()

    def _batch_norm(self, n_features):
        self.use_batch_norm = params['improvements']['use_batch_norm']
        if self.use_batch_norm:
            return nn.BatchNorm2d(n_features)
        return nn.Identity()

    def _initialize_weights(self):
        use_he = params['improvements']['use_he_initialization']
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if use_he:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                else:
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)   
        x = self.fc_layer3(x)   

        return x
