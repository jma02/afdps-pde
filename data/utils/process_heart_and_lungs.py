import h5py
import numpy as np
import torch
import cmocean as cmo
import matplotlib.pyplot as plt
dataset = {}

# Crop from 70x70 to 64x64 (center crop)
crop_size = 64
original_size = 70
margin = (original_size - crop_size) // 2


def print_hdf5_structure(file_path, max_items=20):
    with h5py.File(file_path, 'r') as f:
        print(f"HDF5 file: {file_path}")
        print(f"Root level items: {len(f.keys())}")
        print("-" * 50)
        
        for i, key in enumerate(f.keys()):
            if i >= max_items:
                print(f"... and {len(f.keys()) - max_items} more items")
                break
                
            item = f[key]
            if isinstance(item, h5py.Group):
                print(f"GROUP: {key} (contains {len(item.keys())} items)")
            elif isinstance(item, h5py.Dataset):
                print(f"DATASET: {key} - shape: {item.shape}, dtype: {item.dtype}")
            else:
                print(f"OTHER: {key} - type: {type(item)}")

        name_maps = {
            'training': 'train',
            'testing': 'test',
            'validation': 'val'
        }
        for split in ["training", "testing", "validation"]:
            data_samples = f[split]
            data_outputs= []
            for key in data_samples.keys():
                output_data = data_samples[key]["output"][:]
                cropped_output = output_data[margin:margin+crop_size, margin:margin+crop_size]
                data_outputs.append(cropped_output)
            dataset[name_maps[split]] = torch.tensor(np.array(data_outputs)).to(torch.float32).to('cpu')
        print(dataset['train'].shape)
        print(dataset['test'].shape)
        print(dataset['val'].shape)
        torch.save(dataset, 'data/heart_and_lungs.pt')

        # Preview first few samples in cartesian coordinates
        num_samples = 4
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_samples):
            ax = axes[i]
            sample = dataset['train'][i].numpy()
            ax.imshow(sample, cmap=cmo.cm.balance, vmin=-1, vmax=1)
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')

        # Use our new dataset
        plt.savefig('data/heart_and_lungs_training_samples.png')

# Print the structure

print_hdf5_structure('data/heart_and_lungs/BodyEIT.h5')

