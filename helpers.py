import torch
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np








def label_correction(target, outputs, classes, training_loss):
    # Used to correct labels when epochs > 250
    outputs = np.eye(classes)[np.argmax(outputs, axis=1)]
    W = ((training_loss - np.min(training_loss))/(np.max(training_loss) - np.min(training_loss))).reshape([len(training_loss), 1])
    return W*target + (1-W)*outputs


class Momentum_Model_Optimizer():
    # Momentum model which is updated as a moving average of the training model
    def __init__(self, model, momentum_model):
        self.model = model
        self.model_parameters= [model.state_dict().values()]
        self.momentum_model = momentum_model
        self.momentum_parameters= [momentum_model.state_dict().values()]
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def training_step(self):
        for param, ema_param in zip(self.model_parameters, self.momentum_parameters):
            if param.type()=='torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param = ema_param.mul_(0.999)+(0.001*param)




def get_output(model, device, batch):
    predictions = []
    loss = []
    model.eval().to(device)
    with torch.no_grad:
        for inputs, targets in batch:
            output = model(inputs)
            preds = F.softmax(output, dim=1)
            predictions.append(preds.numpy())
            l = F.cross_entropy(output, targets, reduction='none')
            loss.append(l.numpy())
    return predictions, loss