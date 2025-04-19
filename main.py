import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def mel_filterbank(n_filters=40, n_fft=512, sample_rate=16000, f_min=0, f_max=None):
    if f_max is None:
        f_max = sample_rate / 2

    # Conversion functions between Hz and Mel
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    # Compute Mel frequency points
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin frequencies
    bin_frequencies = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create triangular filters
    filters = torch.zeros(n_filters, n_fft // 2 + 1)

    for i in range(1, n_filters + 1):
        f_left = bin_frequencies[i - 1]
        f_center = bin_frequencies[i]
        f_right = bin_frequencies[i + 1]

        # Rising edge of the filter
        if f_center > f_left:
            filters[i - 1, f_left:f_center] = torch.linspace(0, 1, f_center - f_left)
        # Falling edge of the filter
        if f_right > f_center:
            filters[i - 1, f_center:f_right] = torch.linspace(1, 0, f_right - f_center)

    return filters

class DualFolderDataset(Dataset):
    def __init__(self, folder1, folder2, prefix1='feature_map_', prefix2='spectrogram_', transform1=None, transform2=None):
        """
        Args:
            folder1 (str): Path to the first folder containing .npy files.
            folder2 (str): Path to the second folder containing .npy files.
            prefix1 (str): Prefix for files in folder1.
            prefix2 (str): Prefix for files in folder2.
            transform1 (callable, optional): Optional transform to be applied to data from folder1.
            transform2 (callable, optional): Optional transform to be applied to data from folder2.
        """
        self.folder1 = folder1
        self.folder2 = folder2
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.transform1 = transform1
        self.transform2 = transform2

        # Extract filenames without prefixes (postfix only) for both folders
        self.folder1_files = {
            f[len(prefix1):].replace('.npy', ''): f 
            for f in os.listdir(folder1) if f.startswith(prefix1) and f.endswith('.npy')
        }
        self.folder2_files = {
            f[len(prefix2):].replace('.npy', ''): f 
            for f in os.listdir(folder2) if f.startswith(prefix2) and f.endswith('.npy')
        }
        self.mel_filters = mel_filterbank(40, 256)
        self.mel_filters = self.mel_filters[:, 1:]
        
        # Keep only postfixes that exist in both folders
        self.common_postfixes = sorted(set(self.folder1_files.keys()) & set(self.folder2_files.keys()))

    def __len__(self):
        return len(self.common_postfixes)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: (data1, data2), where data1 and data2 are the loaded .npy arrays.
        """
        if idx < 0 or idx >= len(self.common_postfixes):
            raise IndexError("Index out of range")

        postfix = self.common_postfixes[idx]

        # Load files from both folders
        file1_path = os.path.join(self.folder1, self.folder1_files[postfix])
        file2_path = os.path.join(self.folder2, self.folder2_files[postfix])

        data1 = np.load(file1_path)
        data2 = np.load(file2_path)

        # Apply transformations if provided
        if self.transform1:
            data1 = self.transform1(data1)
        if self.transform2:
            data2 = self.transform2(data2)

        data1 = np.mean(data1[0], axis=0)
        data2 = 10**(data2/10)  # Convert from dB to linear scale
        data2 = self.mel_filters.numpy().dot(data2)
        
        return data1, data2

# Original model implementation
class LinearNoBiasSoftplus(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearNoBiasSoftplus, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Better initialization for audio processing
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.nn.functional.linear(x, torch.nn.functional.softplus(self.weights), bias=None)
        return torch.transpose(x, 1, 2)

# Improved model with structured constraints
class MelConstrainedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MelConstrainedLinear, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.centers = nn.Parameter(torch.linspace(0, in_features-1, out_features))
        self.bandwidths = nn.Parameter(torch.ones(out_features) * in_features/out_features)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, x):
        # Create triangular mel-like filters
        filter_weights = torch.zeros_like(self.weights)
        in_indices = torch.arange(self.weights.size(1)).float()
        
        for i in range(self.weights.size(0)):
            # Create triangular window centered at self.centers[i]
            center = self.centers[i]
            bandwidth = torch.abs(self.bandwidths[i]) + 1.0
            
            # Linear ramp up and down
            dist = torch.abs(in_indices - center)
            filter_shape = torch.clamp(1.0 - dist/bandwidth, min=0.0)
            
            # Apply filter shape to weights
            filter_weights[i] = self.weights[i] * filter_shape
            
        x = torch.transpose(x, 1, 2)
        x = torch.nn.functional.linear(x, torch.nn.functional.softplus(filter_weights), bias=None)
        return torch.transpose(x, 1, 2)

# Predefined mel-basis transform
class MelBasisTransform(nn.Module):
    def __init__(self, in_features, out_features):
        super(MelBasisTransform, self).__init__()
        self.mel_scaling = nn.Parameter(torch.ones(out_features, in_features))
        # Initialize with mel filter bank
        with torch.no_grad():
            # Create approximation of mel filter bank 
            centers = torch.linspace(0, in_features-1, out_features)
            for i in range(out_features):
                center = centers[i]
                width = in_features / out_features * 2
                x = torch.arange(in_features).float()
                self.mel_scaling[i] = torch.exp(-0.5 * ((x - center) / width) ** 2)
            
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.matmul(x, self.mel_scaling.t())
        return torch.transpose(x, 1, 2)

# Function to calculate spectral smoothness loss
def spectral_smoothness_loss(weights, lambda_smooth=0.001):
    # Calculate differences between adjacent frequency bands
    diff = weights[:, 1:] - weights[:, :-1]
    return lambda_smooth * torch.mean(diff**2)

# Function for visualizing filters
def visualize_filters(model, save_path=None):
    if hasattr(model, 'weights'):
        p = torch.nn.functional.softplus(model.weights).detach().cpu().numpy()
    elif hasattr(model, 'mel_scaling'):
        p = model.mel_scaling.detach().cpu().numpy()
    else:
        raise ValueError("Model doesn't have appropriate weights to visualize")
    
    plt.figure(figsize=(15, 10))
    for i in range(min(32, p.shape[1])):
        plt.subplot(8, 4, i+1)
        plt.plot(p[:, i])
        plt.title(f"Filter {i+1}")
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_filters.png")
    plt.show()
    
    # Also visualize as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(p, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Input Feature')
    plt.ylabel('Output Channel')
    plt.title('Weight Pattern Heatmap')
    if save_path:
        plt.savefig(f"{save_path}_heatmap.png")
    plt.show()

# Function to train the model
def train_model(model, dataset, num_epochs=200, batch_size=128, learning_rate=0.001, 
                use_smoothness=True, lambda_smooth=0.001, save_path=None):
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    loss_history = []
    
    for epoch in range(num_epochs):
        global_loss = 0.0
        
        for inp, tar in data_loader:
            # Reset gradients
            optimizer.zero_grad()
            
            # Normalize inputs and targets
            inp = (inp - inp.mean(dim=1, keepdim=True)) / (inp.std(dim=1, keepdim=True) + 1e-8)
            tar = (tar - tar.mean(dim=1, keepdim=True)) / (tar.std(dim=1, keepdim=True) + 1e-8)
            
            # Downsample target to match model output dimensions
            tar = torch.nn.functional.avg_pool1d(tar, 4)
            
            # Forward pass
            out = model(inp)
            
            # Ensure outputs and targets have the same size
            n = min(out.shape[2], tar.shape[2])
            tar = tar[:, :, :n]
            out = out[:, :, :n]
            
            # Calculate loss
            mse_loss = torch.mean((tar - out)**2.0)
            
            # Add smoothness regularization if enabled
            if use_smoothness and hasattr(model, 'weights'):
                smooth_loss = spectral_smoothness_loss(model.weights, lambda_smooth)
                loss = mse_loss + smooth_loss
            else:
                loss = mse_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track loss
            global_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = global_loss / len(data_loader)
        loss_history.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Visualize filters periodically
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            if save_path:
                visualize_filters(model, f"{save_path}_epoch_{epoch+1}")
            else:
                visualize_filters(model)
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}_loss.png")
    plt.show()
    
    return model, loss_history

# Main execution
if __name__ == "__main__":
    # Create dataset
    dataset = DualFolderDataset('extracted_features/feature_maps/', 'extracted_features/spectrograms/')
    
    # Get a sample for visualization
    inp, tar = dataset[3]
    
    # Plot sample data
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(inp, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('Input - Feature Map')
    
    plt.subplot(1, 2, 2)
    plt.imshow(tar, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('Target - Spectrogram')
    plt.tight_layout()
    plt.savefig('sample_data.png')
    plt.show()
    
    # Define model parameters
    in_channels = 32
    out_channels = 40
    
    # Initialize models
    print("Training LinearNoBiasSoftplus model...")
    model1 = LinearNoBiasSoftplus(in_channels, out_channels)
    model1, loss1 = train_model(model1, dataset, num_epochs=200, use_smoothness=True, 
                               lambda_smooth=0.001, save_path='linear_model')
    
    print("Training MelConstrainedLinear model...")
    model2 = MelConstrainedLinear(in_channels, out_channels)
    model2, loss2 = train_model(model2, dataset, num_epochs=200, use_smoothness=True, 
                               lambda_smooth=0.001, save_path='mel_constrained_model')
    
    print("Training MelBasisTransform model...")
    model3 = MelBasisTransform(in_channels, out_channels)
    model3, loss3 = train_model(model3, dataset, num_epochs=200, use_smoothness=False, 
                               save_path='mel_basis_model')
    
    # Compare losses
    plt.figure(figsize=(10, 5))
    plt.plot(loss1, label='LinearNoBiasSoftplus')
    plt.plot(loss2, label='MelConstrainedLinear')
    plt.plot(loss3, label='MelBasisTransform')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_comparison.png')
    plt.show()
    
    # Save models
    torch.save(model1.state_dict(), 'linear_model.pth')
    torch.save(model2.state_dict(), 'mel_constrained_model.pth')
    torch.save(model3.state_dict(), 'mel_basis_model.pth')
    
    # Test models on a sample
    inp, tar = dataset[10]
    inp = torch.tensor(inp).unsqueeze(0)
    tar = torch.tensor(tar).unsqueeze(0)
    tar = torch.nn.functional.avg_pool1d(tar, 4)
    
    # Normalize inputs
    inp = (inp - inp.mean(dim=1, keepdim=True)) / (inp.std(dim=1, keepdim=True) + 1e-8)
    
    # Get predictions
    with torch.no_grad():
        out1 = model1(inp)
        out2 = model2(inp)
        out3 = model3(inp)
    
    # Plot predictions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(tar[0].numpy(), interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('Target Spectrogram')
    
    plt.subplot(2, 2, 2)
    plt.imshow(out1[0].numpy(), interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('LinearNoBiasSoftplus Prediction')
    
    plt.subplot(2, 2, 3)
    plt.imshow(out2[0].numpy(), interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('MelConstrainedLinear Prediction')
    
    plt.subplot(2, 2, 4)
    plt.imshow(out3[0].numpy(), interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('MelBasisTransform Prediction')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()