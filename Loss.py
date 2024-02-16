import torch
import torch.nn as nn
import torchvision.models as models

class total_loss(nn.Module):
    def __init__(self):
        super(total_loss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for i in network.parameters():
            i.requires_grad = False
        self.network = network
        self.mseloss = nn.MSELoss()
        
    def forward(self, label,output, goal):
        adv_loss = torch.mean(1 - label)
        perc_loss = self.mseloss(self.network(output), self.network(goal))
        image_loss = self.mseloss(output, goal)
        return image_loss + 0.006 * perc_loss + 0.001 * adv_loss
