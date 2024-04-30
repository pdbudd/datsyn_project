import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, io, transforms
import pandas as pd
import torch, os
from PIL import Image

class BoundingBoxDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        """
        Args:
            img_dir (string): Directory with all the images.
            annotation_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.img_dir = os.path.join(img_dir,"img1/")
        if train:
            self.annotations = pd.read_csv(os.path.join(img_dir,"gt/gt.txt"))
        #self.unique_images = self.annotations['img_number'].tolist()
        self.unique_images = 1802
        self.transform = transform

    def __len__(self):
        return self.unique_images

    def __getitem__(self, idx):
        # Calculate indices of the frame and its neighbors
        num_neighbors = 1  # Fetch one neighboring frame on each side

        frames = []
        
        # Loop to gather frames
        for i in range(idx - num_neighbors, idx + num_neighbors + 1):
            if i < 1:
                frame_idx = 1  # Pad with the first image if out of lower boundary
            elif i > self.unique_images:
                frame_idx = self.unique_images  # Pad with the last image if out of upper boundary
            else:
                frame_idx = i

            # Format the file name and load the image
            filename = f"{frame_idx:06d}.jpg"
            img_path = os.path.join(self.img_dir, filename)
            image = Image.open(img_path)
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)

            frames.append(image)

        # Stack frames along a new dimension to maintain temporal order
        stacked_frames = torch.stack(frames, dim=0)  # Stacking along new dimension

        # Retrieve annotations for the central frame only
        central_annotations = self.annotations[self.annotations['img_number'] == int(frame_idx)]
        bboxes = torch.tensor(central_annotations[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values, dtype=torch.float32)
        labels = torch.tensor(central_annotations['class'].values, dtype=torch.long)
        if not self.train:
            return stacked_frames
        return {
            'stacked_frames': stacked_frames,  # This is now a tensor of shape [3, C, H, W] assuming 3 channels per image
            'bboxes': bboxes,
            'labels': labels
        }

class RBKDataModule(pl.LightningDataModule):
    def __init__(self, data_root, train_str, transforms=None, frames_per_sample=3, batch_size=50, num_workers=16, train_split_ratio = 0.8):
        """
        Args:
            data_folder (string): Path to the folder where images are stored.
            annotation_file (string): Path to the annotation file.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.data_root = data_root
        self.train_str = train_str
        self.transforms = transforms
        self.frames_per_sample = frames_per_sample
        self.train_split_ratio = train_split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.annotations = pd.read_csv(os.path.join(data_root,train_str,"gt/gt.txt"), names=['img_number', 'feature_ID', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'unknown1', 'class', 'unknown2'])
        
        # Group annotations by img_number and collect all bounding boxes and classes per image
        #self.grouped_annotations = self.annotations.groupby('img_number').apply(lambda x: x[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'class']].values).to_dict()

    def prepare_data(self):
        pass
    
    def prepare_data_per_node(self):
        pass

    def setup(self, stage=None):
        # Use the custom dataset class that you have defined
        full_dataset = BoundingBoxDataset(img_dir=os.path.join(self.data_root,self.train_str),
                                     transform=self.get_transforms("train"))  # Assuming the same transforms for simplicity

        # Randomly shuffle the indices
        indices = range(1,1802)
        #indices = torch.randperm(len(full_dataset))

        # Calculate split sizes
        val_size = int(len(full_dataset) * (1 - self.train_split_ratio))
        train_size = len(full_dataset) - val_size

        # Split the dataset
        self.train_dataset = Subset(full_dataset, indices[:train_size])
        self.val_dataset = Subset(full_dataset, indices[train_size:])
       
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test_path = os.path.join(self.data_root,"3_test_1_min_hamkam_from_start/")
        test_dataset = BoundingBoxDataset(test_path, train=False, transform=self.get_transforms("test"))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
    def get_transforms(self,split):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        shared_transforms = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ]
        
        if split == "train":
            return transforms.Compose([
                *shared_transforms,
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                transforms.RandomHorizontalFlip(),
                # ...
            ])
            
        elif split == "val":
            return transforms.Compose([
                *shared_transforms,
                # ...
            ])
        elif split == "test":
            return transforms.Compose([
                *shared_transforms,
                # ...
            ])