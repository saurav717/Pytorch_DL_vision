from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1     = nn.Conv2d(1, 20, 5, 1)
        init.xavier_uniform_(self.conv1.weight ) 
        self.norm1     = nn.BatchNorm2d(20)
        self.conv2     = nn.Conv2d(20, 50, 5, 1)
        init.xavier_uniform_(self.conv2.weight )
        self.norm2     = nn.BatchNorm2d(50)
        
        self.fc1       = nn.Linear(4*4*50, 500)
        init.xavier_uniform_(self.fc1.weight )
        self.norm3     = nn.BatchNorm1d(500)
        self.fc2       = nn.Linear(500, 10)
        init.xavier_uniform_(self.fc2.weight )
        self.norm4     = nn.BatchNorm1d(10)
        self.dropout   = nn.Dropout(0)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.norm3(self.fc1(x)))
        x = self.dropout(x)
        x = self.norm4(self.fc2(x))
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

        
if __name__ == '__main__':
    main()

#####################################
#
# 4 - A
#
#####################################
# with ReLU
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.007342
#
# Test set: Average loss: 0.0316, Accuracy: 9900/10000 (99%)
#
#####################################
#
# with sigmoid
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.302332
#
# Test set: Average loss: 0.2308, Accuracy: 9317/10000 (93%)
#
######################################
#
# Here it can be clearly seen that ReLU works far better than sigmoid as an activation function
# for the given number of epochs. It maybe because of the reason that the neurons are not firing 
# with greater magnitude when there is a change in an input which is indirectly leading to requiring
# more number of epochs for reaching a local or global minima, so that the neurons fire, when the
# inputs change. ( Between some inputs there maybe small changes and the neurons may not be responding
# to these inputs as they are supposed to be thereby making the error function less steeper for us to
# requiring more time to reach a local minima )
#
######################################






#####################################
#
# 4 - B
#
#####################################
# 
# with dropout = 0.25
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.021079
#
# Test set: Average loss: 0.0258, Accuracy: 9916/10000 (99%) 
#
#####################################
#
# with dropout = 0.5
# 
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.044581
#
# Test set: Average loss: 0.0309, Accuracy: 9900/10000 (99%)
#
######################################
#
# with dropout = 0.75
# 
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.396957
#
# Test set: Average loss: 0.0757, Accuracy: 9806/10000 (98%) 
#
#####################################
#
# with dropout = 1.0
# 
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 2.307389
#
# Test set: Average loss: 1440865.9072, Accuracy: 999/10000 (10%) 
#
#####################################
#
# Dropout is generally used to reduce the overfitting of the neural net. In the given CNN, the training 
# and the testing accuracies are pretty high. Even if we wanted to allege that the neural net is overfitting
# the testing accurary says the other way.
#   
# I do not think that the neural net is overfitting or anything to observe the change for dropout
# 
# As the dropouts values increased, there is a decrease in accuracy that can be observed clearly
# I think it is mostly because, at each iteration one or the other neuron is dropped out 
# and they are not able to learn the right parameters that would lead to the local convergence
# 
######################################







######################################
#
# 4 - C
#
######################################
# 
# with dropout = 0.25
#
# Adding Batch normalization to each layer
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.080672
#
# Test set: Average loss: 0.0310, Accuracy: 9917/10000 (99%)
#
######################################
#
# with no dropout
#
# Adding batch normalization to each layer
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.017237
# 
# Test set: Average loss: 0.0339, Accuracy: 9919/10000 (99%)
#
#######################################
#
# There is not a great change in the test and train accuracies when I remove the dropout for a 
# fixed number of epochs. Although, when I remove the dropout there is a minute increase in accuracy 
# 
# When I add batch normalization along with dropout = 0.25, there is no big change when I did not include
# batch normalization. ( Dropout = 0.25 with and without batch normalization had no big difference in performance )
#
######################################







######################################
#
# 4 - D
#
######################################
# 
# with kaiming normal initialization ( "fan in ")
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.021430
#
# Test set: Average loss: 0.0371, Accuracy: 9908/10000 (99%)
#
######################################
#
# with kaiming uniform ( " fan in ")
# 
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.024918
#
# Test set: Average loss: 0.0384, Accuracy: 9900/10000 (99%)
#
#######################################
#
# with kaiming uniform ( " fan out ")
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.023213
#
# Test set: Average loss: 0.0334, Accuracy: 9916/10000 (99%)
#
#######################################
# 
# with kaiming normal initialization ( "fan out ")
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.013772
#
# Test set: Average loss: 0.0329, Accuracy: 9923/10000 (99%)
#
#######################################
#######################################
# 
# with xavier normal initialization ( " gain = 1 ")
#
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.010094
#
# Test set: Average loss: 0.0311, Accuracy: 9929/10000 (99%)
#
#######################################
#
# with xavier uniform initialization ( " gain = 1 ")
# 
# Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.023888
#
# Test set: Average loss: 0.0300, Accuracy: 9923/10000 (99%)
#
########################################
#
# There is no big difference in the predictions of both Kaiming and xavier initializations, 
#
# However, there is a slight increase in the number of predictions, in xavier  and kaiming initializations,
# when compared to random initialization
#
# The number of correct predictions is a bit more consistant and better in Xavier initialization thank kaiming
#
######################################
