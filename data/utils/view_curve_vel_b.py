import torch

data = torch.load('data/curve_vel_b.pt', map_location='cpu', weights_only=False)
data_train = data['train']
data_test = data['test']
data_val = data['val']
print(f"Training data min, training data max: {data_train.min()}, {data_train.max()}")
print(f"Validation data min, validation data max: {data_val.min()}, {data_val.max()}")
print(f"Test data min, test data max: {data_test.min()}, {data_test.max()}")