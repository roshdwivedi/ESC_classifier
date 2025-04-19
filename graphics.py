import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from main import LinearNoBiasSoftplus, MelConstrainedLinear, MelBasisTransform, DualFolderDataset
import glob
import shutil
from collections import defaultdict

# Set up matplotlib for better visualizations
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory structure
def setup_output_directories():
    """Create organized directory structure for results"""
    # Main results directory
    base_dir = "activation_analysis_results"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Create subdirectories for each model and analysis type
    models = ["LinearNoBiasSoftplus", "MelConstrainedLinear", "MelBasisTransform"]
    analysis_types = ["filter_bank", "activations", "class_analysis", "correlation"]
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/comparison", exist_ok=True)
    
    for model in models:
        for analysis in analysis_types:
            os.makedirs(f"{base_dir}/{model}/{analysis}", exist_ok=True)
    
    return base_dir

def load_models():
    """Load the three pre-trained models"""
    in_channels = 32
    out_channels = 40
    
    # Initialize models with the same architecture
    model1 = LinearNoBiasSoftplus(in_channels, out_channels)
    model2 = MelConstrainedLinear(in_channels, out_channels)
    model3 = MelBasisTransform(in_channels, out_channels)
    
    # Load saved weights
    model1.load_state_dict(torch.load('linear_model.pth'))
    model2.load_state_dict(torch.load('mel_constrained_model.pth'))
    model3.load_state_dict(torch.load('mel_basis_model.pth'))
    
    # Set models to evaluation mode
    model1.eval()
    model2.eval()
    model3.eval()
    
    return model1, model2, model3

def load_class_labels():
    """
    Load class labels from filenames or create synthetic ones if not available
    
    In a real scenario, this would load actual class labels from a file.
    For this example, we'll extract class information from filenames or create synthetic labels.
    """
    try:
        # Try to find class labels from feature map filenames
        files = glob.glob('extracted_features/feature_maps/feature_map_*.npy')
        class_labels = {}
        
        for file in files:
            # Extract class information from filename
            filename = os.path.basename(file)
            # Assuming filename format is feature_map_XXXX_classname.npy
            parts = filename.replace('feature_map_', '').replace('.npy', '').split('_')
            
            if len(parts) >= 2:
                sample_id = parts[0]
                class_name = '_'.join(parts[1:])
                class_labels[sample_id] = class_name
        
        if not class_labels:
            raise FileNotFoundError("Could not extract class labels from filenames")
            
        return class_labels
        
    except (FileNotFoundError, IndexError):
        # If real class labels can't be found, create synthetic ones for demonstration
        print("Creating synthetic class labels for demonstration purposes")
        
        # Create 5 synthetic classes
        synthetic_classes = ["speech", "music", "noise", "animal_sounds", "machinery"]
        
        # Get all sample IDs
        files = glob.glob('extracted_features/feature_maps/feature_map_*.npy')
        sample_ids = [os.path.basename(f).replace('feature_map_', '').replace('.npy', '').split('_')[0]
                     for f in files]
        
        # Assign synthetic classes
        class_labels = {}
        for i, sample_id in enumerate(sample_ids):
            class_labels[sample_id] = synthetic_classes[i % len(synthetic_classes)]
            
        return class_labels

def get_filter_activations(model, dataset, class_labels):
    """
    Analyze filter activations across samples, grouped by class
    
    Args:
        model: The neural network model
        dataset: Dataset containing samples
        class_labels: Dictionary mapping sample IDs to class labels
    
    Returns:
        class_activations: Dictionary of activations grouped by class
        all_activations: All activations across samples
        all_inputs: All input features
        sample_info: List of (sample_id, class) tuples for each processed sample
    """
    model.eval()
    all_activations = []
    all_inputs = []
    sample_info = []
    class_activations = defaultdict(list)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            inp, tar = dataset[i]
            inp_tensor = torch.tensor(inp).unsqueeze(0)
            
            # Get sample ID from dataset
            sample_id = dataset.common_postfixes[i]
            
            # Get class label for this sample
            if sample_id in class_labels:
                class_label = class_labels[sample_id]
            else:
                # Assign default class if not found
                class_label = f"unknown_class_{i % 5}"
            
            # Normalize input as in training
            inp_tensor = (inp_tensor - inp_tensor.mean(dim=1, keepdim=True)) / (inp_tensor.std(dim=1, keepdim=True) + 1e-8)
            
            # Forward pass to get activations
            if hasattr(model, 'weights'):
                # For LinearNoBiasSoftplus and MelConstrainedLinear
                weights = torch.nn.functional.softplus(model.weights)
                
                # Calculate activations across frequency bands
                x = torch.transpose(inp_tensor, 1, 2)
                activation = torch.matmul(x, weights.t())
                activation = torch.transpose(activation, 1, 2)
                
            elif hasattr(model, 'mel_scaling'):
                # For MelBasisTransform
                weights = model.mel_scaling
                
                # Calculate activations
                x = torch.transpose(inp_tensor, 1, 2)
                activation = torch.matmul(x, weights.t())
                activation = torch.transpose(activation, 1, 2)
            
            # Store activations and input
            activation_numpy = activation.squeeze(0).numpy()
            all_activations.append(activation_numpy)
            all_inputs.append(inp)
            sample_info.append((sample_id, class_label))
            
            # Group by class
            class_activations[class_label].append(activation_numpy)
    
    return class_activations, all_activations, all_inputs, sample_info

def get_filter_visualizations(model):
    """Extract and prepare filter visualizations from the model"""
    if hasattr(model, 'weights'):
        return torch.nn.functional.softplus(model.weights).detach().cpu().numpy()
    elif hasattr(model, 'mel_scaling'):
        return model.mel_scaling.detach().cpu().numpy()
    else:
        raise ValueError("Model doesn't have appropriate weights to visualize")

def plot_filter_bank(model, model_name, output_dir):
    """Plot filter bank for the given model"""
    filters = get_filter_visualizations(model)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(16, filters.shape[0])):
        plt.subplot(4, 4, i+1)
        plt.plot(filters[i, :])
        plt.title(f"Filter {i+1}")
        plt.xlabel("Input Feature Index")
        plt.ylabel("Weight Value")
        plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f"{model_name} Filter Bank", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{output_dir}/{model_name}/filter_bank/individual_filters.png", dpi=300)
    plt.close()
    
    # Also plot as heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(filters, cmap='viridis', xticklabels=5, yticklabels=5)
    plt.xlabel("Input Feature Index")
    plt.ylabel("Filter Index")
    plt.title(f"{model_name} Filter Bank Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}/filter_bank/filter_heatmap.png", dpi=300)
    plt.close()

def plot_activation_heatmap(activations, inputs, sample_info, model_name, output_dir):
    """Plot activation heatmap for specific samples with class labels"""
    num_samples = min(5, len(activations))
    
    plt.figure(figsize=(20, 4*num_samples))
    
    for i in range(num_samples):
        sample_id, class_label = sample_info[i]
        
        # Plot input features
        plt.subplot(num_samples, 2, i*2+1)
        plt.imshow(inputs[i], aspect='auto', interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Feature Value')
        plt.title(f"Sample {sample_id}: Class={class_label}")
        plt.xlabel("Time Frame")
        plt.ylabel("Feature Channel")
        
        # Plot corresponding activations
        plt.subplot(num_samples, 2, i*2+2)
        act_data = activations[i]
        plt.imshow(act_data, aspect='auto', interpolation='nearest', cmap='plasma')
        plt.colorbar(label='Activation Value')
        plt.title(f"Filter Activations for {class_label}")
        plt.xlabel("Time Frame")
        plt.ylabel("Filter Index")
    
    plt.tight_layout()
    plt.suptitle(f"{model_name} Sample Activations by Class", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{output_dir}/{model_name}/activations/sample_activations.png", dpi=300)
    plt.close()

def plot_class_activations(class_activations, model_name, output_dir):
    """Plot average activations for each class"""
    # Compute average activation per filter for each class
    class_avg_activations = {}
    for class_name, activations in class_activations.items():
        # Calculate average activation across time and samples
        class_avg_activations[class_name] = np.mean([act.mean(axis=1) for act in activations], axis=0)
    
    # Plot average activation for each class
    plt.figure(figsize=(15, 10))
    for i, (class_name, avg_activations) in enumerate(class_avg_activations.items()):
        plt.subplot(len(class_avg_activations), 1, i+1)
        plt.bar(range(len(avg_activations)), avg_activations)
        plt.title(f"Class: {class_name}")
        plt.xlabel("Filter Index")
        plt.ylabel("Avg Activation")
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.suptitle(f"{model_name} - Average Filter Activations by Class", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{output_dir}/{model_name}/class_analysis/class_average_activations.png", dpi=300)
    plt.close()
    
    # Create activation heatmap across classes
    plt.figure(figsize=(12, 8))
    class_act_matrix = np.vstack([v for v in class_avg_activations.values()])
    ax = sns.heatmap(class_act_matrix, cmap='viridis', 
                     yticklabels=list(class_avg_activations.keys()),
                     xticklabels=range(0, class_act_matrix.shape[1], 5))
    plt.xlabel("Filter Index")
    plt.ylabel("Class")
    plt.title(f"{model_name} - Class-Filter Activation Map")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}/class_analysis/class_filter_heatmap.png", dpi=300)
    plt.close()
    
    # Find the most discriminative filters for each class
    discriminative_filters = {}
    all_filters = np.array(list(class_avg_activations.values()))
    
    for i, (class_name, avg_activations) in enumerate(class_avg_activations.items()):
        # Calculate how much this class's activation differs from others for each filter
        other_classes = np.delete(all_filters, i, axis=0)
        other_mean = np.mean(other_classes, axis=0)
        
        # Difference between this class and average of other classes
        diff = avg_activations - other_mean
        
        # Get top 5 most discriminative filters
        top_filters = np.argsort(-diff)[:5]
        discriminative_filters[class_name] = top_filters
    
    # Plot top discriminative filters for each class
    plt.figure(figsize=(15, len(discriminative_filters)*3))
    
    for i, (class_name, top_filters) in enumerate(discriminative_filters.items()):
        plt.subplot(len(discriminative_filters), 1, i+1)
        
        # Plot activations for all classes for the top filters
        for j, (c_name, activations) in enumerate(class_avg_activations.items()):
            values = activations[top_filters]
            x_pos = np.arange(len(top_filters))
            if c_name == class_name:
                plt.bar(x_pos, values, alpha=0.8, label=c_name)
            else:
                plt.bar(x_pos, values, alpha=0.3, label=c_name)
                
        plt.title(f"Top Discriminative Filters for {class_name}")
        plt.xlabel("Filter Index")
        plt.ylabel("Activation")
        plt.xticks(np.arange(len(top_filters)), top_filters)
        if i == 0:  # Add legend only to the first subplot
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f"{model_name} - Discriminative Filters by Class", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{output_dir}/{model_name}/class_analysis/discriminative_filters.png", dpi=300)
    plt.close()
    
    return discriminative_filters

def plot_feature_filter_correlation(model, dataset, class_labels, model_name, output_dir):
    """Visualize correlation between input features and filter activations"""
    # Get model filters
    filters = get_filter_visualizations(model)
    
    # Gather input features from multiple samples, grouped by class
    class_features = defaultdict(list)
    
    for i in range(len(dataset)):
        inp, _ = dataset[i]
        
        # Get sample ID and class
        sample_id = dataset.common_postfixes[i]
        if sample_id in class_labels:
            class_name = class_labels[sample_id]
        else:
            class_name = f"unknown_class_{i % 5}"
            
        class_features[class_name].append(inp)
    
    # Calculate correlation between features and filters for each class
    class_correlations = {}
    
    for class_name, features in class_features.items():
        # Stack features and compute mean across time dimension
        feature_means = np.array([f.mean(axis=1) for f in features])
        avg_features = np.mean(feature_means, axis=0)
        
        # Calculate correlation between features and filters
        correlation_matrix = np.zeros((filters.shape[0], len(avg_features)))
        
        for i in range(filters.shape[0]):  # For each filter
            for j in range(len(avg_features)):  # For each feature
                # Correlation between filter i and feature j
                correlation_matrix[i, j] = np.corrcoef(filters[i, :], avg_features)[0, 1]
        
        class_correlations[class_name] = correlation_matrix
    
    # Plot correlation matrix for each class
    for class_name, correlation in class_correlations.items():
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, cmap='coolwarm', center=0, 
                    xticklabels=5, yticklabels=5)
        plt.title(f"{model_name} - {class_name} Filter-Feature Correlation")
        plt.xlabel("Input Feature Index")
        plt.ylabel("Filter Index")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}/correlation/{class_name}_correlation.png", dpi=300)
        plt.close()
    
    # Plot average correlation across all classes
    avg_correlation = np.mean([corr for corr in class_correlations.values()], axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_correlation, cmap='coolwarm', center=0, 
                xticklabels=5, yticklabels=5)
    plt.title(f"{model_name} - Average Filter-Feature Correlation")
    plt.xlabel("Input Feature Index")
    plt.ylabel("Filter Index")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}/correlation/average_correlation.png", dpi=300)
    plt.close()
    
    return class_correlations

def plot_model_comparison(model1, model2, model3, dataset, class_labels, output_dir):
    """Compare activations across all three models on the same samples with class information"""
    # Get sample data
    samples = []
    sample_info = []
    
    for i in range(min(5, len(dataset))):
        inp, _ = dataset[i]
        samples.append(inp)
        
        # Get sample ID and class
        sample_id = dataset.common_postfixes[i]
        if sample_id in class_labels:
            class_name = class_labels[sample_id]
        else:
            class_name = f"unknown_class_{i % 5}"
            
        sample_info.append((sample_id, class_name))
    
    # Setup figure
    plt.figure(figsize=(20, 15))
    
    # For a few samples
    for sample_idx in range(min(3, len(samples))):
        sample_id, class_name = sample_info[sample_idx]
        inp = samples[sample_idx]
        inp_tensor = torch.tensor(inp).unsqueeze(0)
        inp_tensor = (inp_tensor - inp_tensor.mean(dim=1, keepdim=True)) / (inp_tensor.std(dim=1, keepdim=True) + 1e-8)
        
        # Get outputs from all models
        with torch.no_grad():
            out1 = model1(inp_tensor).squeeze(0).numpy()
            out2 = model2(inp_tensor).squeeze(0).numpy()
            out3 = model3(inp_tensor).squeeze(0).numpy()
        
        # Plot input
        plt.subplot(3, 4, sample_idx*4+1)
        plt.imshow(inp, aspect='auto', interpolation='nearest', cmap='viridis')
        plt.title(f"Sample {sample_id}: Class={class_name}")
        plt.xlabel("Time Frame")
        plt.ylabel("Feature Channel")
        plt.colorbar()
        
        # Plot outputs from each model
        plt.subplot(3, 4, sample_idx*4+2)
        plt.imshow(out1, aspect='auto', interpolation='nearest', cmap='plasma')
        plt.title("LinearNoBiasSoftplus Output")
        plt.xlabel("Time Frame")
        plt.ylabel("Filter Channel")
        plt.colorbar()
        
        plt.subplot(3, 4, sample_idx*4+3)
        plt.imshow(out2, aspect='auto', interpolation='nearest', cmap='plasma')
        plt.title("MelConstrainedLinear Output")
        plt.xlabel("Time Frame")
        plt.ylabel("Filter Channel")
        plt.colorbar()
        
        plt.subplot(3, 4, sample_idx*4+4)
        plt.imshow(out3, aspect='auto', interpolation='nearest', cmap='plasma')
        plt.title("MelBasisTransform Output")
        plt.xlabel("Time Frame")
        plt.ylabel("Filter Channel")
        plt.colorbar()
    
    plt.tight_layout()
    plt.suptitle("Model Comparison: Activations by Class", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.savefig(f"{output_dir}/comparison/activation_comparison.png", dpi=300)
    plt.close()

def compare_class_specific_activations(model1, model2, model3, class_discriminative_filters, output_dir):
    """Compare how the different models activate on class-specific patterns"""
    # Get filter banks
    filters1 = get_filter_visualizations(model1)
    filters2 = get_filter_visualizations(model2)
    filters3 = get_filter_visualizations(model3)
    
    # For each class, compare the top discriminative filters across models
    for class_name, filter_indices in class_discriminative_filters['LinearNoBiasSoftplus'].items():
        plt.figure(figsize=(15, 10))
        
        # Plot the top discriminative filters for each model
        for i, filter_idx in enumerate(filter_indices[:3]):  # Show top 3 filters
            plt.subplot(3, 3, i*3+1)
            plt.plot(filters1[filter_idx])
            plt.title(f"LinearNoBiasSoftplus\nFilter {filter_idx}")
            plt.xlabel("Input Feature Index")
            plt.ylabel("Weight")
            plt.grid(True)
            
            # Find most similar filter in model2 by correlation
            correlations = []
            for j in range(filters2.shape[0]):
                corr = np.corrcoef(filters1[filter_idx], filters2[j])[0, 1]
                correlations.append(corr)
            best_match = np.argmax(correlations)
            
            plt.subplot(3, 3, i*3+2)
            plt.plot(filters2[best_match])
            plt.title(f"MelConstrainedLinear\nFilter {best_match} (corr={correlations[best_match]:.2f})")
            plt.xlabel("Input Feature Index")
            plt.ylabel("Weight")
            plt.grid(True)
            
            # Find most similar filter in model3 by correlation
            correlations = []
            for j in range(filters3.shape[0]):
                corr = np.corrcoef(filters1[filter_idx], filters3[j])[0, 1]
                correlations.append(corr)
            best_match = np.argmax(correlations)
            
            plt.subplot(3, 3, i*3+3)
            plt.plot(filters3[best_match])
            plt.title(f"MelBasisTransform\nFilter {best_match} (corr={correlations[best_match]:.2f})")
            plt.xlabel("Input Feature Index")
            plt.ylabel("Weight")
            plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f"Class {class_name} - Top Discriminative Filters Across Models", fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.savefig(f"{output_dir}/comparison/class_{class_name}_filters.png", dpi=300)
        plt.close()

def generate_html_report(output_dir):
    """Generate an HTML report summarizing all findings"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Network Filter Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; margin-top: 30px; }
            h3 { color: #2980b9; }
            .image-container { margin: 20px 0; }
            .image-container img { max-width: 100%; border: 1px solid #ddd; }
            .model-section { margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
            .comparison-section { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }
            .filter-analysis { display: flex; flex-wrap: wrap; }
            .filter-card { margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; flex: 1; min-width: 300px; }
        </style>
    </head>
    <body>
        <h1>Neural Network Filter Analysis Report</h1>
        <p>This report analyzes the behavior of three different neural network models designed to transform feature maps into spectrogram-like representations. The analysis focuses on how different filters activate for various audio classes and compares the models' approaches.</p>
    """
    
    # Add sections for each model
    models = ["LinearNoBiasSoftplus", "MelConstrainedLinear", "MelBasisTransform"]
    for model in models:
        html_content += f"""
        <div class="model-section">
            <h2>{model} Model Analysis</h2>
            
            <h3>Filter Bank Visualization</h3>
            <div class="image-container">
                <img src="{model}/filter_bank/filter_heatmap.png" alt="{model} filter heatmap">
                <p>Heatmap showing the learned filters. X-axis represents input feature index, Y-axis shows filter index.</p>
            </div>
            
            <div class="image-container">
                <img src="{model}/filter_bank/individual_filters.png" alt="{model} individual filters">
                <p>Individual filter responses showing frequency sensitivity patterns.</p>
            </div>
            
            <h3>Class-specific Activations</h3>
            <div class="image-container">
                <img src="{model}/class_analysis/class_filter_heatmap.png" alt="{model} class-filter heatmap">
                <p>Heatmap showing how each filter activates for different classes. Brighter areas indicate stronger activation.</p>
            </div>
            
            <div class="image-container">
                <img src="{model}/class_analysis/discriminative_filters.png" alt="{model} discriminative filters">
                <p>The most discriminative filters for each class. These filters show the strongest activation patterns that differentiate classes.</p>
            </div>
            
            <h3>Sample Activations</h3>
            <div class="image-container">
                <img src="{model}/activations/sample_activations.png" alt="{model} sample activations">
                <p>Activation patterns for specific samples from different classes.</p>
            </div>
            
            <h3>Feature-Filter Correlations</h3>
            <div class="image-container">
                <img src="{model}/correlation/average_correlation.png" alt="{model} correlation">
                <p>Correlation between input features and filter activations, showing which input features influence which filters.</p>
            </div>
        </div>
        """
    
    # Add model comparison section
    html_content += """
        <div class="comparison-section">
            <h2>Model Comparison</h2>
            
            <div class="image-container">
                <img src="comparison/activation_comparison.png" alt="Model activation comparison">
                <p>Comparison of how each model activates on the same input samples.</p>
            </div>
            
            <h3>Class-specific Filter Comparison</h3>
    """
    
    # Add class-specific comparisons if available
    class_images = glob.glob(f"{output_dir}/comparison/class_*_filters.png")
    for image in class_images:
        class_name = os.path.basename(image).replace("class_", "").replace("_filters.png", "")
        html_content += f"""
            <div class="image-container">
                <img src="comparison/class_{class_name}_filters.png" alt="{class_name} filter comparison">
                <p>Comparison of discriminative filters for class '{class_name}' across models.</p>
            </div>
        """
    
    html_content += """
        </div>
        
        <h2>Conclusion</h2>
        <p>This analysis demonstrates how different neural network architectures learn to transform feature maps into spectrograms, highlighting the different filter patterns each model develops and how they respond to different audio classes.</p>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(f"{output_dir}/analysis_report.html", "w") as f:
        f.write(html_content)
    
    print(f"HTML report generated at {output_dir}/analysis_report.html")

def run_analysis():
    """Main function to run all analyses"""
    print("Setting up output directories...")
    output_dir = setup_output_directories()
    
    print("Loading models...")
    model1, model2, model3 = load_models()
    
    print("Loading dataset and class labels...")
    dataset = DualFolderDataset('extracted_features/feature_maps/', 'extracted_features/spectrograms/')
    class_labels = load_class_labels()
    
    # Visualize filter banks
    print("Visualizing filter banks...")
    plot_filter_bank(model1, "LinearNoBiasSoftplus", output_dir)
    plot_filter_bank(model2, "MelConstrainedLinear", output_dir)
    plot_filter_bank(model3, "MelBasisTransform", output_dir)
    
    # Get activations for each model with class information
    print("Analyzing class-specific activations for LinearNoBiasSoftplus...")
    class_act1, all_act1, inputs1, sample_info1 = get_filter_activations(model1, dataset, class_labels)
    plot_activation_heatmap(all_act1, inputs1, sample_info1, "LinearNoBiasSoftplus", output_dir)
    disc_filters1 = plot_class_activations(class_act1, "LinearNoBiasSoftplus", output_dir)
    
    print("Analyzing class-specific activations for MelConstrainedLinear...")
    class_act2, all_act2, inputs2, sample_info2 = get_filter_activations(model2, dataset, class_labels)
    plot_activation_heatmap(all_act2, inputs2, sample_info2, "MelConstrainedLinear", output_dir)
    disc_filters2 = plot_class_activations(class_act2, "MelConstrainedLinear", output_dir)
    
    print("Analyzing class-specific activations for MelBasisTransform...")
    print("Analyzing class-specific activations for MelBasisTransform...")
    class_act3, all_act3, inputs3, sample_info3 = get_filter_activations(model3, dataset, class_labels)
    plot_activation_heatmap(all_act3, inputs3, sample_info3, "MelBasisTransform", output_dir)
    disc_filters3 = plot_class_activations(class_act3, "MelBasisTransform", output_dir)
    
    # Plot feature-filter correlations
    print("Analyzing feature-filter correlations for LinearNoBiasSoftplus...")
    corr1 = plot_feature_filter_correlation(model1, dataset, class_labels, "LinearNoBiasSoftplus", output_dir)
    
    print("Analyzing feature-filter correlations for MelConstrainedLinear...")
    corr2 = plot_feature_filter_correlation(model2, dataset, class_labels, "MelConstrainedLinear", output_dir)
    
    print("Analyzing feature-filter correlations for MelBasisTransform...")
    corr3 = plot_feature_filter_correlation(model3, dataset, class_labels, "MelBasisTransform", output_dir)
    
    # Compare models
    print("Generating model comparisons...")
    plot_model_comparison(model1, model2, model3, dataset, class_labels, output_dir)
    
    # Store discriminative filters for all models
    class_discriminative_filters = {
        "LinearNoBiasSoftplus": disc_filters1,
        "MelConstrainedLinear": disc_filters2,
        "MelBasisTransform": disc_filters3
    }
    
    # Compare class-specific activations across models
    print("Comparing class-specific patterns across models...")
    compare_class_specific_activations(model1, model2, model3, class_discriminative_filters, output_dir)
    
    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"View the full report at {output_dir}/analysis_report.html")

if __name__ == "__main__":
    run_analysis()