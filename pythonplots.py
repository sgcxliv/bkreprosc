import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Set the overall style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Load the data exported from R
data = pd.read_csv("bkgam_smooth_data.csv")

# Extract unique models and measures
models = data['model'].unique()
measures = data['measure'].unique()

# Define the color palette - using a viridis color map for model types
# This creates a visually appealing and colorblind-friendly palette
model_palette = sns.color_palette("viridis", len(models))
model_colors = dict(zip(models, model_palette))

# Function to create a separate plot for each model and measure
def create_separate_plots(data, output_dir="./plots", figsize=(8, 6), dpi=300):
    """
    Create a separate plot file for each model and measure combination.
    
    Parameters:
    -----------
    data : DataFrame
        The GAM plot data
    output_dir : str
        Directory to save plots to
    figsize : tuple
        Figure size for each plot
    dpi : int
        Resolution for saving figures
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each model-measure combination separately
    for model in models:
        for measure in measures:
            # Filter data for this model and measure
            subset = data[(data['model'] == model) & (data['measure'] == measure)]
            
            if subset.empty:
                print(f"No data for {model} - {measure}, skipping...")
                continue
            
            # Create a new figure for this combination
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot the smooth curve
            ax.plot(subset['x'], subset['y'], color=model_colors[model],
                    linewidth=2.5, label=model)
            
            # Add confidence interval
            ax.fill_between(subset['x'], subset['lower'], subset['upper'],
                           color=model_colors[model], alpha=0.3)
            
            # Set titles and labels
            ax.set_title(f"{model} - {measure}")
            ax.set_xlabel(f"{measure}")
            ax.set_ylabel("Reading Time (ms)")
            
            # Add a light grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Create a filename-safe version of the model name
            safe_model = model.replace(' ', '_').replace('-', '_')
            
            # Save the figure
            filename = f"{output_dir}/{safe_model}_{measure}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            
            print(f"Saved plot to {filename}")
    
    print(f"Created {len(models) * len(measures)} individual plots in {output_dir}")

# Function to create separate comparison plots for each measure
def create_separate_measure_plots(data, output_dir="./plots", figsize=(10, 8), dpi=300):
    """
    Create separate plots for each measure, showing all models together.
    
    Parameters:
    -----------
    data : DataFrame
        The GAM plot data
    output_dir : str
        Directory to save plots to
    figsize : tuple
        Figure size for each plot
    dpi : int
        Resolution for saving figures
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a separate plot for each measure
    for measure in measures:
        # Create a new figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each model for this measure
        for model in models:
            subset = data[(data['model'] == model) & (data['measure'] == measure)]
            
            if subset.empty:
                continue
                
            ax.plot(subset['x'], subset['y'], color=model_colors[model],
                    linewidth=2, label=model)
        
        # Set titles and labels
        ax.set_title(f"All Models - {measure}")
        ax.set_xlabel(f"{measure}")
        ax.set_ylabel("Reading Time (ms)")
        
        # Add legend
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        filename = f"{output_dir}/all_models_{measure}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        
        print(f"Saved comparison plot to {filename}")

# Function to create separate plots for model groups
def create_model_group_plots(data, output_dir="./plots", figsize=(10, 8), dpi=300):
    """
    Create separate plots for groups of related models.
    
    Parameters:
    -----------
    data : DataFrame
        The GAM plot data
    output_dir : str
        Directory to save plots to
    figsize : tuple
        Figure size for each plot
    dpi : int
        Resolution for saving figures
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model groups
    model_groups = {
        "human_cloze": [m for m in models if "Cloze" in m],
        "gpt_models": [m for m in models if "GPT" in m],
        "llama_models": [m for m in models if "LLaMA" in m],
        "top_llms": [m for m in models if any(llm in m for llm in ["GPT-NeoX", "LLaMA-2", "OLMO-2"])]
    }
    
    # Create plots for each group and measure
    for group_name, group_models in model_groups.items():
        if not group_models:  # Skip if no models in this group
            continue
            
        for measure in measures:
            # Create a new figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot each model in this group
            for model in group_models:
                subset = data[(data['model'] == model) & (data['measure'] == measure)]
                
                if subset.empty:
                    continue
                    
                ax.plot(subset['x'], subset['y'], color=model_colors[model],
                        linewidth=2.5, label=model)
                ax.fill_between(subset['x'], subset['lower'], subset['upper'],
                               color=model_colors[model], alpha=0.2)
            
            # Set titles and labels
            group_display_name = group_name.replace('_', ' ').title()
            ax.set_title(f"{group_display_name} - {measure}")
            ax.set_xlabel(f"{measure}")
            ax.set_ylabel("Reading Time (ms)")
            
            # Add legend
            ax.legend(loc='best')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save the figure
            filename = f"{output_dir}/{group_name}_{measure}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            
            print(f"Saved group plot to {filename}")

# Create all the separate plots
create_separate_plots(data)
create_separate_measure_plots(data)
create_model_group_plots(data)

print("All plotting complete!")
