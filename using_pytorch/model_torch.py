import torch 
import torch.nn as nn 
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        # We optimize dropout rate in a convolutional neural network.
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.batch1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch7 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch8 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool9 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=25088, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=3, bias=True) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        
        x = self.conv7(x)
        x = self.batch7(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        x = self.batch8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        x = self.pool9(x)
        x = x.view(x.size(0), -1) 
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x) 
        return F.log_softmax(x, dim=-1)


#net = Net()
#x = torch.randn(2, 3, 256, 256)
#y = net(x)
#print(y.size())