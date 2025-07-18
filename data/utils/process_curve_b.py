import torch 
import numpy as np
from einops import rearrange
import glob
import os

device = 'cpu'

# Load all model files
model_files = sorted(glob.glob('data/curve_vel_b/model/model*.npy'))
print(f"Found {len(model_files)} model files")

# Load all data
data_list = []
for file in model_files:
    data = np.load(file)
    print(f"Loaded {file}: {data.shape}")
    data_list.append(data)

stacked_data = np.stack(data_list, axis=0)
print(f"Stacked data shape: {stacked_data.shape}")

# Reshape from (60, 500, 1, 70, 70) to (60*500, 1, 70, 70)
reshaped_data = rearrange(stacked_data, 'n b c h w -> (n b) c h w')
print(f"Reshaped data shape: {reshaped_data.shape}")


# Convert to tensor
data_tensor = torch.from_numpy(reshaped_data).to(device)
print(f"Final tensor shape: {data_tensor.shape}")

# Crop from 70x70 to 64x64 (center crop)
# Calculate crop margins
crop_size = 64
original_size = 70
margin = (original_size - crop_size) // 2

# Crop the data: [batch, channels, height, width]
data_tensor = data_tensor[:, :, margin:margin+crop_size, margin:margin+crop_size]
print(f"Cropped tensor shape: {data_tensor.shape}")


# Split into train/val/test sets
total_samples = data_tensor.shape[0]
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_size = int(total_samples * train_ratio)
val_size = int(total_samples * val_ratio)
test_size = total_samples - train_size - val_size

# Create random indices
indices = torch.randperm(total_samples)

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split data
train_data = data_tensor[train_indices]
val_data = data_tensor[val_indices]
test_data = data_tensor[test_indices]

print(f"Train set shape: {train_data.shape}")
print(f"Validation set shape: {val_data.shape}")
print(f"Test set shape: {test_data.shape}")

# Save the splits
dataset = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}


torch.save(dataset, 'data/curve_vel_b.pt')
print("Dataset saved to data/curve_vel_b.pt")