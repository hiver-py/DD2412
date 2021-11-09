import torch
import torch.nn.functional as F

def training(model, optimizer, momentum_optimizer, sigma, batches):
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
        acc += prediction.eq(target.view(prediction.size())).sum().item()
    accuracy = acc/n
    loss = l/n
    return accuracy, loss

def testing(model, batches):
    device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    l, acc = 0
    n = batches.size(0)
    model.eval()
    with torch.nograd():
        for input, target in batches:
            output = model(input)          
            l += F.cross_entropy(output, target, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)
            acc +=  prediction.eq(target.view(prediction.size())).sum().item()
    accuracy = acc/n
    return accuracy
