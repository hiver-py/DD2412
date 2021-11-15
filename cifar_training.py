import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from network.wideresnet import Wide_ResNet
from eval import training, testing
from helpers import label_correction, Momentum_Model_Optimizer, get_output


batch_size = 128
epochs = 300
lr = 0.001
momentum = 0.9
wd = 5e-4
# Sigma for CIFAR10 is 1 for symmetric noise, 0.5 otherwise
# Sigma for CIFAR100 is 0.1 for instance-dependent noise, and 0.2 otherwise
# Sigma is set to 0 during label correction phase
sigma = 1
device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)


# Read data 

def read_data(dataset):
    if dataset == "CIFAR100":
        raise NotImplementedError
    else:
        raise NotImplementedError





# Model

wide_resnet, momentum_model = Wide_ResNet().cuda()
[parameter.detach() for parameter in momentum_model.parameters()]


cudnn.benchmark = True
optimizer = optim.SGD(wide_resnet.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
momentum_optimizer = Momentum_Model_Optimizer(wide_resnet, momentum_model)


# Training

for epoch in range(1, epochs+1):
    

    if epoch > 250:
        sigma = 0
        print("hello")


    train_accuracy, train_loss = training(wide_resnet, optimizer, momentum_optimizer, sigma, data)
    testing_accuracy, test_loss = testing(momentum_model, data)


exp_name = 'sigma{:.1f}_{}_{}{:.1f}_seed{}'.format(dataset,noise_mode,noise_rate,sigma) 
torch.save(wide_resnet.state_dict(), '{}/net.pth'.format(exp_name))
torch.save(momentum_model.state_dict(), '{}/ema_net.pth'.format(exp_name))