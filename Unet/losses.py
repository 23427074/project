from __future__ import print_function, division
import torch.nn.functional as F
import torch
import torch.nn as nn


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()
    # print("intersection: ", intersection)

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def bce_dice_loggss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
   #  print("bce: ", bce)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)
    # print("calc_loss_dice: ", dice)

    loss = bce * bce_weight + dice * (1 - bce_weight)
   # print("loss: ", loss)

    return loss

class bce_dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(bce_dice_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
            
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
                                     
        return Dice_BCE
                                        
class focal_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(focal_loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.3, gamma=2, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
                                                    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
                                
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss



class tversky_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(tversky_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.3, beta=0.7):
                                    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
                                 
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
                                                                        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
                                                                                        
        return 1 - Tversky


class focal_tversky_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(focal_tversky_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.8, beta=0.2, gamma=2):
                                    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
                                                    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
                                                                                    
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  

        FocalTversky = (1 - Tversky)**gamma                                         
        
        return FocalTversky


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds
