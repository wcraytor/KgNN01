# For comments, refer to the Jupyter version under the notebooks folder.

import ssl
import urllib.request
import certifi

# Multiple SSL bypass approaches
ssl._create_default_https_context = ssl._create_unverified_context


# Also set urllib's default context
import urllib.request
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from datetime import datetime

os.chdir('/Volumes/Nvme_1/Desktop/github/KgNN01')

class RoomPhotoDataset:
    """
    Manages room photos organized in images/<room>/ folders
    """

    def __init__(self, base_path='images', max_photos_per_room=None, sample_percentage=100, random_seed=42):
        """
        Initialize the room dataset manager

        Args:
            base_path: Path to images folder containing room subfolders
            max_photos_per_room: Maximum number of photos per room (None for all)
            sample_percentage: Percentage of photos to use (1-100)
            random_seed: For reproducible splits and sampling
        """
        self.base_path = base_path
        self.max_photos_per_room = max_photos_per_room
        self.sample_percentage = max(1, min(100, sample_percentage))  # Clamp between 1-100
        self.random_seed = random_seed

        # Room types
        self.room_types = [
            'bathroom', 'bedroom', 'dining', 'gaming', 'kitchen',
            'laundry', 'living', 'office', 'terrace', 'yard'
        ]

        # Create room to index mapping
        self.room_to_idx = {room: idx for idx, room in enumerate(self.room_types)}
        self.idx_to_room = {idx: room for room, idx in self.room_to_idx.items()}

        # Dataset size configurations
        self.size_configs = {
            'tiny': 100,
            'small': 500,
            'medium': 1000,
            'large': 2000,
            'xl': 5000,
            'full': None  # Use all available
        }

        # Split ratios
        self.split_ratios = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }

        # Load and organize all photos
        self.photo_metadata = self._load_photo_metadata()
        self._create_splits()

    def _load_photo_metadata(self):
        """Load all photo paths and labels from room folders with optional sampling"""
        # Check if base path exists
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Images directory not found: {self.base_path}")

        photos = []
        room_counts = {}
        total_available = 0
        total_sampled = 0

        # Set random seed for consistent sampling
        np.random.seed(self.random_seed)

        print(f"Sampling {self.sample_percentage}% of available photos...")

        for room_idx, room_type in enumerate(self.room_types):
            room_path = os.path.join(self.base_path, room_type)

            if not os.path.exists(room_path):
                print(f"Warning: Room folder not found: {room_path}")
                room_counts[room_type] = 0
                continue

            # Get all photos in this room folder
            all_room_photos = []
            for file in os.listdir(room_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    full_path = os.path.join(room_path, file)
                    all_room_photos.append({
                        'path': full_path,
                        'filename': file,
                        'room_type': room_type,
                        'label': room_idx,
                        'room_folder': room_type
                    })

            total_available += len(all_room_photos)

            # Apply sampling percentage
            if self.sample_percentage < 100:
                sample_size = max(1, int(len(all_room_photos) * self.sample_percentage / 100))
                room_photos = np.random.choice(all_room_photos, size=sample_size, replace=False).tolist()
                print(f"  {room_type}: {len(all_room_photos)} available → {len(room_photos)} sampled ({self.sample_percentage}%)")
            else:
                room_photos = all_room_photos
                print(f"  {room_type}: {len(room_photos)} photos (100%)")

            # Apply max_photos_per_room if specified
            if self.max_photos_per_room and len(room_photos) > self.max_photos_per_room:
                room_photos = np.random.choice(room_photos, self.max_photos_per_room, replace=False).tolist()
                print(f"    Limited to {self.max_photos_per_room} photos per room")

            photos.extend(room_photos)
            room_counts[room_type] = len(room_photos)
            total_sampled += len(room_photos)

        # Check if we found any photos
        if len(photos) == 0:
            raise ValueError(f"No image files found in {self.base_path}. "
                           f"Make sure room folders contain image files.")

        # Convert to DataFrame
        df = pd.DataFrame(photos)

        print(f"\nDataset Summary:")
        print(f"Total photos available: {total_available}")
        print(f"Total photos sampled: {total_sampled} ({self.sample_percentage}%)")
        print(f"Photos loaded into memory: {len(df)}")
        print("\nRoom distribution after sampling:")
        for room, count in room_counts.items():
            print(f"  {room}: {count} photos")

        return df

    def _create_splits(self):
        """Create stratified train/val/test splits maintaining room balance"""
        np.random.seed(self.random_seed)

        # Stratified split to maintain room balance across splits
        train_val, test = train_test_split(
            self.photo_metadata,
            test_size=self.split_ratios['test'],
            stratify=self.photo_metadata['label'],
            random_state=self.random_seed
        )

        train, val = train_test_split(
            train_val,
            test_size=self.split_ratios['val'] / (1 - self.split_ratios['test']),
            stratify=train_val['label'],
            random_state=self.random_seed
        )

        # Add split column
        self.photo_metadata.loc[train.index, 'split'] = 'train'
        self.photo_metadata.loc[val.index, 'split'] = 'val'
        self.photo_metadata.loc[test.index, 'split'] = 'test'

        print("\nSplit distribution:")
        split_counts = self.photo_metadata.groupby(['split', 'room_type']).size().unstack(fill_value=0)
        print(split_counts)

    def get_dataset_subset(self, size='full', split='train'):
        """
        Get a specific subset of the data

        Args:
            size: 'tiny', 'small', 'medium', 'large', 'xl', or 'full'
            split: 'train', 'val', or 'test'

        Returns:
            DataFrame with the requested subset
        """
        if size not in self.size_configs:
            raise ValueError(f"Size must be one of {list(self.size_configs.keys())}")

        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        # Get photos for this split
        split_photos = self.photo_metadata[self.photo_metadata['split'] == split].copy()

        # If using full dataset, return all
        if size == 'full' or self.size_configs[size] is None:
            subset = split_photos
        else:
            # Calculate how many photos we need for this size and split
            total_size = self.size_configs[size]
            split_ratio = self.split_ratios[split]
            target_count = int(total_size * split_ratio)

            # Sample stratified by room type to maintain balance
            if len(split_photos) >= target_count:
                subset = split_photos.groupby('room_type', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, target_count // len(self.room_types))),
                                     random_state=self.random_seed)
                ).head(target_count)
            else:
                subset = split_photos

        print(f"\n{size.title()} {split} set: {len(subset)} photos")
        room_dist = subset['room_type'].value_counts().to_dict()
        print(f"Room distribution: {room_dist}")

        return subset.reset_index(drop=True)

class RoomDataset(Dataset):
    """PyTorch Dataset for loading room photos"""

    def __init__(self, photo_metadata, transform=None):
        """
        Args:
            photo_metadata: DataFrame with 'path' and 'label' columns
            transform: torchvision transforms to apply
        """
        self.photo_metadata = photo_metadata
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.photo_metadata)

    def __getitem__(self, idx):
        row = self.photo_metadata.iloc[idx]

        # Load and resize image (1200x1016 -> center crop -> 512x512)
        try:
            image = Image.open(row['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (512, 512), color='white')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, row['label']

def get_device():
    """
    Get the best available device for training
    Prioritizes: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")

    return device

def create_optimized_transforms():
    """Create optimized transforms for better accuracy"""

    # Training transforms with aggressive augmentation
    train_transform = transforms.Compose([
        # Start with your 1200x1016 images
        transforms.Resize(600),  # Resize to reasonable size first
        transforms.CenterCrop(512),  # Center crop to 512x512 (minimal loss)

        # Advanced augmentation for better accuracy
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomGrayscale(p=0.05),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(600),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform

def create_dataloaders(dataset_manager, size='full', batch_size=16, num_workers=0):
    """
    Create train/val/test dataloaders for room classification

    Args:
        dataset_manager: RoomPhotoDataset instance
        size: Dataset size to use
        batch_size: Batch size for dataloaders (optimized for 64GB RAM)
        num_workers: Number of workers for data loading

    Returns:
        dict with 'train', 'val', 'test' dataloaders
    """

    # Get optimized transforms
    train_transform, val_test_transform = create_optimized_transforms()

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        # Get subset
        subset_metadata = dataset_manager.get_dataset_subset(size, split)

        # Choose transform
        transform = train_transform if split == 'train' else val_test_transform

        # Create dataset
        dataset = RoomDataset(subset_metadata, transform=transform)

        # Create dataloader (optimized for 64GB memory)
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # Don't pin memory for MPS
            persistent_workers=num_workers > 0
        )

    return dataloaders

class OptimizedRoomNet(nn.Module):
    """Transfer learning model for room classification with optimizations"""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(OptimizedRoomNet, self).__init__()

        # Use pre-trained ResNet50 for transfer learning
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze early layers for faster training and better generalization
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, epochs=20, learning_rate=0.001, device=None):
    """
    Train the model with optimizations and return training history
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    # Use AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'epochs': [], 'train_time': [], 'learning_rates': []
    }

    print(f"Training for {epochs} epochs on {device}")
    print("-" * 60)

    best_val_acc = 0.0
    best_model_state = None

    # Initialize epoch statistics log
    epoch_log_lines = []
    epoch_log_lines.append(f"Epoch Statistics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    epoch_log_lines.append("=" * 80)
    epoch_log_lines.append(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12} {'Time(s)':<8}")
    epoch_log_lines.append("-" * 80)

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Step the scheduler
        scheduler.step()

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        history['train_time'].append(epoch_time)
        history['learning_rates'].append(current_lr)

        # Add to epoch log
        epoch_log_line = f"{epoch+1:<6} {avg_train_loss:<12.4f} {train_acc:<12.2f} {avg_val_loss:<12.4f} {val_acc:<12.2f} {current_lr:<12.6f} {epoch_time:<8.1f}"
        epoch_log_lines.append(epoch_log_line)

        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

    total_time = sum(history['train_time'])
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Add summary to epoch log
    epoch_log_lines.append("")
    epoch_log_lines.append("=" * 80)
    epoch_log_lines.append(f"Training Summary:")
    epoch_log_lines.append(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    epoch_log_lines.append(f"Best validation accuracy: {best_val_acc:.2f}%")
    epoch_log_lines.append(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    epoch_log_lines.append(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")

    # Save epoch statistics to file
    with open('EpochStatistics.log', 'w') as f:
        f.write('\n'.join(epoch_log_lines))

    print("Epoch statistics saved to EpochStatistics.log")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")

    return history, best_val_acc

def evaluate_model_and_save_results(model, test_loader, dataset_manager, device, save_path='room_predictions.txt'):
    """
    Evaluate model on test set and save predictions vs actual
    """
    model.eval()

    all_predictions = []
    all_actuals = []
    all_paths = []

    # Track per-room accuracy
    room_correct = defaultdict(int)
    room_total = defaultdict(int)

    test_correct = 0
    test_total = 0

    print("Evaluating model on test set...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = output.max(1)

            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

            # Store predictions and actuals
            batch_predictions = predicted.cpu().numpy()
            batch_actuals = target.cpu().numpy()

            all_predictions.extend(batch_predictions)
            all_actuals.extend(batch_actuals)

            # Track per-room accuracy
            for pred, actual in zip(batch_predictions, batch_actuals):
                room_name = dataset_manager.idx_to_room[actual]
                room_total[room_name] += 1
                if pred == actual:
                    room_correct[room_name] += 1

    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Convert indices to room names
    predicted_rooms = [dataset_manager.idx_to_room[idx] for idx in all_predictions]
    actual_rooms = [dataset_manager.idx_to_room[idx] for idx in all_actuals]

    # Calculate per-room accuracies
    room_accuracies = {}
    for room in dataset_manager.room_types:
        if room_total[room] > 0:
            room_accuracies[room] = 100. * room_correct[room] / room_total[room]
        else:
            room_accuracies[room] = 0.0

    # Create detailed report
    report_lines = []
    report_lines.append(f"Room Classification Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append(f"Test Accuracy: {test_acc:.2f}%")
    report_lines.append(f"Total Test Images: {len(all_predictions)}")
    report_lines.append("")

    # Classification report
    report_lines.append("Detailed Classification Report:")
    report_lines.append("-" * 40)
    class_report = classification_report(actual_rooms, predicted_rooms, target_names=dataset_manager.room_types)
    report_lines.append(class_report)
    report_lines.append("")

    # Confusion matrix
    report_lines.append("Confusion Matrix:")
    report_lines.append("-" * 20)
    cm = confusion_matrix(actual_rooms, predicted_rooms, labels=dataset_manager.room_types)

    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=dataset_manager.room_types, columns=dataset_manager.room_types)
    report_lines.append(str(cm_df))
    report_lines.append("")

    # Individual predictions (first 100 for brevity)
    report_lines.append("Sample Predictions (first 100):")
    report_lines.append("-" * 35)
    report_lines.append(f"{'Actual':<12} {'Predicted':<12} {'Correct'}")
    report_lines.append("-" * 35)

    for i in range(min(100, len(actual_rooms))):
        correct = "✓" if actual_rooms[i] == predicted_rooms[i] else "✗"
        report_lines.append(f"{actual_rooms[i]:<12} {predicted_rooms[i]:<12} {correct}")

    # Save main results to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Results saved to {save_path}")

    # Save per-room test results to separate file
    room_results_lines = []
    room_results_lines.append(f"Test Results by Room Type - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    room_results_lines.append("=" * 60)
    room_results_lines.append(f"Overall Test Accuracy: {test_acc:.2f}%")
    room_results_lines.append("")
    room_results_lines.append("Per-Room Test Accuracy:")
    room_results_lines.append("-" * 40)
    room_results_lines.append(f"{'Room Type':<15} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    room_results_lines.append("-" * 40)

    for room in sorted(dataset_manager.room_types):
        correct = room_correct[room]
        total = room_total[room]
        accuracy = room_accuracies[room]
        room_results_lines.append(f"{room:<15} {correct:<8} {total:<8} {accuracy:<10.2f}%")

    room_results_lines.append("")
    room_results_lines.append("Room Performance Summary:")
    room_results_lines.append("-" * 30)

    # Sort rooms by accuracy
    sorted_rooms = sorted(room_accuracies.items(), key=lambda x: x[1], reverse=True)
    room_results_lines.append("Best performing rooms:")
    for room, acc in sorted_rooms[:3]:
        room_results_lines.append(f"  {room}: {acc:.2f}%")

    room_results_lines.append("")
    room_results_lines.append("Worst performing rooms:")
    for room, acc in sorted_rooms[-3:]:
        room_results_lines.append(f"  {room}: {acc:.2f}%")

    # Save room-specific results
    with open('../TestResultsByRoomType.log', 'w') as f:
        f.write('\n'.join(room_results_lines))

    print("Per-room test results saved to TestResultsByRoomType.log")

    # Also create confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Room Classification Confusion Matrix\nTest Accuracy: {test_acc:.2f}%')
    plt.ylabel('Actual Room')
    plt.xlabel('Predicted Room')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('room_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return test_acc

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history with optimizations"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    epochs = history['epochs']

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate plot
    ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)

    # Training time per epoch
    ax4.bar(epochs, history['train_time'], alpha=0.7, color='orange')
    ax4.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_room_classifier(base_path='images', dataset_size='full', epochs=20, batch_size=16, sample_percentage=100):
    """
    Complete training pipeline for room classification

    Args:
        base_path: Path to images folder
        dataset_size: 'tiny', 'small', 'medium', 'large', 'xl', or 'full'
        epochs: Number of training epochs
        batch_size: Batch size for training
        sample_percentage: Percentage of available photos to use (1-100)
    """
    print("="*80)
    print("ROOM CLASSIFICATION TRAINING PIPELINE")
    print("="*80)

    # Initialize dataset with sampling
    print("Loading dataset...")
    dataset_manager = RoomPhotoDataset(base_path=base_path, sample_percentage=sample_percentage)

    # Create dataloaders
    print(f"\nCreating dataloaders for {dataset_size} dataset...")
    dataloaders = create_dataloaders(dataset_manager, size=dataset_size, batch_size=batch_size, num_workers=0)

    # Get device
    device = get_device()

    # Create optimized model
    print("\nCreating optimized model...")
    model = OptimizedRoomNet(num_classes=len(dataset_manager.room_types))

    # Train model
    print("\nStarting training...")
    history, best_val_acc = train_model(
        model, dataloaders['train'], dataloaders['val'],
        epochs=epochs, learning_rate=0.001, device=device
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc = evaluate_model_and_save_results(
        model, dataloaders['test'], dataset_manager, device,
        save_path='../room_classification_results.txt'
    )

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, save_path='room_training_history.png')


    print(f"\nTraining completed!")
    print(f"Dataset sampling: {sample_percentage}% of available photos")
    print(f"Total photos used: {len(dataset_manager.photo_metadata)}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Results saved to: room_classification_results.txt")
    print(f"Per-room results: TestResultsByRoomType.log")
    print(f"Epoch statistics: EpochStatistics.log")
    print(f"Plots saved to: room_training_history.png, room_confusion_matrix.png")

    return model, history, dataset_manager

# Example usage
if __name__ == "__main__":
    # Run complete training pipeline
    # Adjust batch_size based on your 64GB memory - start with 16
    model, history, dataset_manager = train_room_classifier(
        base_path='/Volumes/Nvme_1/Desktop/github/KgNN01/images/',
        dataset_size='full',  # Use all photos
        epochs=20,
        batch_size=16,  # Optimized for 64GB memory
        sample_percentage=10
    )

    print("\nRoom types learned:")
    for i, room in enumerate(dataset_manager.room_types):
        print(f"  {i}: {room}")