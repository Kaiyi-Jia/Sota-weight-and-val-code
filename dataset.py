import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter


class FERDataset(Dataset):
    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split  # 'train', 'val', or 'test'
        self.transform = transform if transform else self._default_transform(split)
        self.data, self.labels = self._load_data()

    def _default_transform(self, split):
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_data(self):
        if self.config.dataset == 'rafdb':
            return self._load_rafdb()
        elif self.config.dataset == 'ferplus':
            return self._load_ferplus()
        elif self.config.dataset in ['affectnet7', 'affectnet8']:
            return self._load_affectnet()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

    def _load_rafdb(self):
        root = self.config.data_root
        # Attempt to locate the RAF-DB subdirectory
        rafdb_root = os.path.join(root, 'RAF-DB') if os.path.exists(os.path.join(root, 'RAF-DB')) else root

        # Prefer official format (presence of {split}_label.txt file)
        label_file = os.path.join(rafdb_root, f'{self.split}_label.txt')
        if os.path.exists(label_file):
            print(f"Loading RAF-DB in official format (label file: {label_file})")
            return self._load_rafdb_official(rafdb_root)
        else:
            print(f"Loading RAF-DB in folder-per-class format from {rafdb_root}/{self.split}")
            return self._load_rafdb_folder_per_class(rafdb_root)

    def _load_rafdb_official(self, root):
        # Official format: Images in Image/aligned/{split}, labels in {split}_label.txt
        img_dir = os.path.join(root, 'Image', 'aligned', self.split)
        label_file = os.path.join(root, f'{self.split}_label.txt')
        if not os.path.exists(img_dir):
            # Compatibility for test and val mapping (if split is val but directory is test)
            if self.split == 'val' and os.path.exists(os.path.join(root, 'Image', 'aligned', 'test')):
                img_dir = os.path.join(root, 'Image', 'aligned', 'test')
            elif self.split == 'test' and os.path.exists(os.path.join(root, 'Image', 'aligned', 'val')):
                img_dir = os.path.join(root, 'Image', 'aligned', 'val')
        df = pd.read_csv(label_file, sep=' ', header=None, names=['file', 'label'])
        data, labels = [], []
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['file'])
            if os.path.exists(img_path):
                data.append(img_path)
                labels.append(row['label'] - 1)  # 1-based to 0-based
        return data, labels

    def _load_rafdb_folder_per_class(self, root):
        # Folder-per-class format: root/{split}/0/, root/{split}/1/, ...
        split_dir = os.path.join(root, self.split)
        if not os.path.exists(split_dir):
            # Fallback if split is val but directory is test
            if self.split == 'val' and os.path.exists(os.path.join(root, 'test')):
                split_dir = os.path.join(root, 'test')
            elif self.split == 'test' and os.path.exists(os.path.join(root, 'val')):
                split_dir = os.path.join(root, 'val')
            else:
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
        data, labels = [], []
        for class_idx in range(self.config.class_num):
            class_dir = os.path.join(split_dir, str(class_idx))
            if not os.path.isdir(class_dir):
                print(f"Warning: class directory {class_dir} not found, skipping class {class_idx}")
                continue
            img_paths = glob.glob(os.path.join(class_dir, '*.*'))
            img_paths = [p for p in img_paths if p.lower().endswith(('jpg', 'jpeg', 'png'))]
            for p in img_paths:
                data.append(p)
                labels.append(class_idx)
        return data, labels

    def _load_ferplus(self):
        root = self.config.data_root
        # Automatically locate FER+ directory, typically named 'FERPlus' or 'ferplus'
        ferplus_root = os.path.join(root, 'FERPlus') if os.path.exists(os.path.join(root, 'FERPlus')) else root

        print(f"Loading FER+ in folder-per-class format from {ferplus_root}/{self.split}")
        return self._load_rafdb_folder_per_class(ferplus_root)

    def _load_affectnet(self):
        root = self.config.data_root

        # Attempt to distinguish folder names: Some might be split into 'AffectNet7' and 'AffectNet8', others just 'AffectNet'
        folder_name = 'AffectNet7' if self.config.dataset == 'affectnet7' else 'AffectNet8'

        if os.path.exists(os.path.join(root, folder_name)):
            affectnet_root = os.path.join(root, folder_name)
        elif os.path.exists(os.path.join(root, 'AffectNet')):
            affectnet_root = os.path.join(root, 'AffectNet')
        else:
            affectnet_root = root

        print(f"Loading {self.config.dataset.upper()} in folder-per-class format from {affectnet_root}/{self.split}")

        # Reuse the folder-per-class reading logic
        # If config.dataset is 'affectnet7', it will automatically read folders 0~6
        # If config.dataset is 'affectnet8', it will automatically read folders 0~7 (including Contempt)
        return self._load_rafdb_folder_per_class(affectnet_root)

    def get_class_counts(self, num_classes):
        counts = Counter(self.labels)
        return [counts.get(i, 0) for i in range(num_classes)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_dataloaders(config, num_classes):
    train_set = FERDataset(config, split='train')
    val_set = FERDataset(config, split='val')

    train_class_counts = train_set.get_class_counts(num_classes)

    # 2. Build DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_class_counts
