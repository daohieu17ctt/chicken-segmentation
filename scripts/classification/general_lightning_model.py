
import pytorch_lightning as pl
import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision import models, transforms


class ChickenClassification(pl.LightningModule):
    """
    """
    def __init__(self, model, class_weight=None):
        super().__init__()
        self.model = model
        self.class_weights = class_weight

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss_total =  F.cross_entropy(out,y)
        self.log('train_loss', loss_total, logger=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        weight_val = torch.tensor(self.class_weights).to(
            x.device) if self.class_weights else None
        loss = F.cross_entropy(logits, y, weight=weight_val) 
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss)
        return {'val_loss': loss, 'val_acc': acc}
        #return {'val_loss': loss, 'labels': y, 'preds': preds}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #preds = torch.cat([x['preds'] for x in outputs])
        #labels = torch.cat([x['labels'] for x in outputs])
        #acc = accuracy(preds, labels)
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_log = {'val_loss': avg_loss}

        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_loss', avg_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)

        return {'labels': y, 'preds': preds}

    def test_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])
        # paths = torch.cat([x['paths'] for x in outputs])

        acc = accuracy(preds, labels)

        return {
            'test_acc': acc
        }

    def configure_optimizers(self):
        """
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=3e-4)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=5),
            'interval': 'epoch', # The unit of the scheduler's step size",
            'frequency': 1, # The frequency of the scheduler",
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler",
            'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor",
            'strict': True, # Whether to crash the training if `monitor` is not found",
        }
        return [optimizer], [scheduler]
