import torch    
import torch.nn.functional as F

def Make_Optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

def Make_LR_Scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def Make_Loss_Function(number_of_classes):
    class DiceCELoss:
        def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
            self.weight = weight
            self.epsilon = epsilon
            self.mode = mode

        @staticmethod
        def ce2d_deterministic(logits, target, weight=None, ignore_index=-100, eps=1e-7):
            log_p = torch.log_softmax(logits, dim=1)
            log_p = log_p.gather(1, target.unsqueeze(1))
            log_p = log_p.squeeze(1)

            mask = (target != ignore_index)
            if weight is not None:
                w = weight[target]
                loss = -(w * log_p * mask).sum() / (w * mask).sum()
            else:
                loss = -(log_p * mask).sum() / mask.sum()

            return loss

        
        def __call__(self, pred, target):
            if self.mode == 'binary':
                pred = pred.squeeze(1)
                target = target.squeeze(1).float()
                intersection = torch.sum(pred * target, dim=(1, 2))
                union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                
                ce_loss = F.binary_cross_entropy(pred, target)
                combined_loss = 0.5 * dice_loss + 0.5 * ce_loss
            
            elif self.mode == 'multiclass':
                batchsize, num_classes, H, W = pred.shape
                pred_soft = torch.softmax(pred, dim=1)
                target = target.squeeze(1)
                target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
                intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
                union = torch.sum(pred_soft, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()

                ce_loss = DiceCELoss.ce2d_deterministic(pred, target)
                combined_loss = 0.5 * dice_loss + 0.5 * ce_loss
            else:
                raise ValueError("mode should be 'binary' or 'multiclass'")
            
            return combined_loss
    
    BINARY_SEG = True if number_of_classes==2 else False
    return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass')
