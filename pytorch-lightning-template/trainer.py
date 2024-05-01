from datamodule_1 import RBKDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path

torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

def bbox_acc(box1, box2):
    """
    Compute IoU (Intersection over Union) of multiple pairs of bounding boxes.

    Args:
        box1 (torch.Tensor): Tensor of bounding box coordinates [x1, y1, w, h] for multiple boxes.
        box2 (torch.Tensor): Tensor of bounding box coordinates [x1, y1, w, h] for multiple boxes.

    Returns:
        float: Average IoU value for all pairs of boxes.
    """
    # Reshape box tensors to have shape (num_boxes, 4) if necessary
    if len(box1.shape) == 1:
        box1 = box1.view(-1, 4)
    if len(box2.shape) == 1:
        box2 = box2.view(-1, 4)

    # Compute coordinates of intersection rectangle
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 0] + box1[:, 2], box2[:, 0] + box2[:, 2])
    y2 = torch.min(box1[:, 1] + box1[:, 3], box2[:, 1] + box2[:, 3])

    # Compute area of intersection rectangle
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute areas of bounding boxes
    box1_area = box1[:, 2] * box1[:, 3]
    box2_area = box2[:, 2] * box2[:, 3]

    # Compute union area
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    # Compute average IoU across all pairs of boxes
    avg_iou = torch.mean(iou)

    return avg_iou.item()  # Convert to Python float

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        weights = ResNet50_Weights.DEFAULT if config.use_pretrained_weights else None
        self.model = resnet50(weights=weights)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Determine the number of layers in the model
        total_layers = len(list(self.model.children()))

        # Freeze all layers except the last 3
        for idx, child in enumerate(self.model.children()):
            if idx < total_layers - 4:
                for param in child.parameters():
                    param.requires_grad = False

        self.bbox_head = nn.Linear(num_features, 4*self.config.num_boxes)
        self.bbox_loss_fn = torch.nn.SmoothL1Loss()
        self.bbox_class = nn.Linear(num_features, self.config.num_boxes)
        self.class_loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = self.Accuracy
    
    def Accuracy(self, bbox_preds, bbox_labels, class_preds, class_labels):
        accuracy = 0
        if torch.argmax(class_preds) == torch.argmax(class_labels):
            accuracy += 0.15
        
        bbox_accuracy = bbox_acc(bbox_preds, bbox_labels)
        accuracy += 0.85*bbox_accuracy
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x):
        features = self.model(x)
        bbox_preds = self.bbox_head(features)
        class_preds = self.bbox_class(features)
        return bbox_preds, class_preds

    def training_step(self, batch, batch_idx):
        stacked_frames = batch['stacked_frames']
        labels = batch['labels']
        bboxes = batch['bboxes']
        bbox_preds, class_preds = self.forward(stacked_frames)
        # Compute losses
        bbox_loss = self.bbox_loss_fn(bbox_preds, bboxes)
        class_loss = self.class_loss_fn(class_preds, labels)
        
        # Combine losses
        total_loss = bbox_loss + class_loss
        
        acc = self.acc_fn(bbox_preds, bboxes, class_preds, labels)
        
        self.log_dict({
            "train/loss": total_loss,
            "train/bbox_loss": bbox_loss,
            "train/class_loss": class_loss,
            "train/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        stacked_frames = batch['stacked_frames']
        labels = batch['labels']
        bboxes = batch['bboxes']
        
        bbox_preds, class_preds = self.forward(stacked_frames)
        # Compute losses
        bbox_loss = self.bbox_loss_fn(bbox_preds, bboxes)
        class_loss = self.class_loss_fn(class_preds, labels)
        
        # Combine losses
        total_loss = bbox_loss + class_loss
        
        acc = self.acc_fn(bbox_preds, bboxes, class_preds, labels)
        
        self.log_dict({
            "val/loss": total_loss,
            "val/bbox_loss": bbox_loss,
            "val/class_loss": class_loss,
            "val/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        stacked_frames = batch['stacked_frames']
        labels = batch['labels']
        bboxes = batch['bboxes']
        
        bbox_preds, class_preds = self.forward(stacked_frames)
        
        # Compute losses
        bbox_loss = self.bbox_loss_fn(bbox_preds, bboxes)
        class_loss = self.class_loss_fn(class_preds, labels)
        
        # Combine losses
        total_loss = bbox_loss + class_loss
        
        acc = self.acc_fn(bbox_preds, bboxes, class_preds, labels)
        
        self.log_dict({
            "test/loss": total_loss,
            "test/bbox_loss": bbox_loss,
            "test/class_loss": class_loss,
            "test/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return total_loss

        

if __name__ == "__main__":
    
    pl.seed_everything(42)
    
    dm = RBKDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        frames_per_sample=config.frames_per_sample,
        train_split_ratio=config.train_split_ratio,
        data_root=config.data_root,
        train_str=config.train_data
    )
    if config.checkpoint_path:
        model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
    else:
        model = LitModel(config)

    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)
