import torch
import torch.nn.functional as F

def training(model, classes, optimizer, momentum_optimizer, sigma, batches):
    device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    l, acc = 0
    model.train()
    n = batches.size(0)
    for input, target in batches:
        input = input.to(device) 
        target = target.to(device)
        target += sigma*torch.randn(target.size())
        output = model(input)
        prediction = output.argmax(dim=1, keepdim=True)
        l = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        momentum_optimizer.step()
        l += input.size(0)*l.item()
        acc = prediction.eq(target.view(prediction.size())).sum().item()
    accuracy = acc/n
    loss = l/n
    return accuracy, loss


def train(args, model, classes, optimizer, ema_optimizer, device, loader):
    model.train()
    train_loss = 0
    correct = 0

    for data, target in loader:
        
        if len(target.size())==1:
            target = torch.zeros(target.size(0), classes).scatter_(1, target.view(-1,1), 1) 

        data, target = data.to(device), target.to(device)
            
        # SLN
        if args.sigma>0:
            target += args.sigma*torch.randn(target.size()).to(device)
        
        output = model(data)
        loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_optimizer:
            ema_optimizer.step()
        
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        if len(target.size())==2: # soft target
            target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    return train_loss/len(loader.dataset), correct/len(loader.dataset)