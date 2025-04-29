import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Load the data exported from R
setwd("/afs/cs.stanford.edu/u/sgcxliv/")
smooth_data = pd.read_csv("gam_smooth_data.csv")
raw_data = pd.read_csv("gam_raw_data.csv")

# Create model type and predictor columns
smooth_data['model_type'] = smooth_data['model_name'].apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
smooth_data['predictor_type'] = smooth_data['model_name'].apply(lambda x: x.split(' ')[-1])

raw_data['model_type'] = raw_data['model_name'].apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
raw_data['predictor_type'] = raw_data['model_name'].apply(lambda x: x.split(' ')[-1])

# Get unique model types and predictor types
model_types = sorted(smooth_data['model_type'].unique())
predictor_types = sorted(smooth_data['predictor_type'].unique())

# Create a color map for different model types
colors = sns.color_palette("viridis", len(model_types))
color_map = dict(zip(model_types, colors))

# Create a pretty plot of all models
def create_model_plots(fig_size=(20, 28), show_raw_data=True, alpha_raw=0.05):
    """Create a grid of plots for all models, with one plot per model-predictor combination."""
    
    # Calculate the number of rows needed (one row per model type)
    n_rows = len(model_types)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 2, figsize=fig_size,
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.4})
    
    # If there's only one model type, make axes 2D
    if n_rows == 1:
        axes = np.array([axes])
        
    # Iterate through model types
    for i, model_type in enumerate(model_types):
        # Filter data for this model type
        model_smooth = smooth_data[smooth_data['model_type'] == model_type]
        model_raw = raw_data[raw_data['model_type'] == model_type]
        
        # Get the color for this model
        color = color_map[model_type]
        
        # Plot each predictor type
        for j, pred_type in enumerate(predictor_types):
            # Further filter by predictor type
            pred_smooth = model_smooth[model_smooth['predictor_type'] == pred_type]
            pred_raw = model_raw[model_raw['predictor_type'] == pred_type]
            
            # Skip if no data
            if len(pred_smooth) == 0:
                continue
            
            # Plot raw data points if requested
            if show_raw_data:
                axes[i, j].scatter(pred_raw['x'], pred_raw['y'],
                                  color=color, alpha=alpha_raw, s=10)
            
            # Plot the smooth line
            axes[i, j].plot(pred_smooth['x'], pred_smooth['y'],
                           color=color, linewidth=2.5)
            
            # Add confidence intervals
            axes[i, j].fill_between(pred_smooth['x'],
                                   pred_smooth['lower'],
                                   pred_smooth['upper'],
                                   color=color, alpha=0.2)
            
            # Set labels
            axes[i, j].set_xlabel(f'{model_type} {pred_type}')
            axes[i, j].set_ylabel('Reading Time (ms)')
            axes[i, j].set_title(f'{model_type} {pred_type} vs RT')
    
    # Add a figure title
    fig.suptitle('Relationship Between Predictability Measures and Reading Times',
                fontsize=24, y=0.98)
    
    # Add a legend for the models
    legend_elements = [Line2D([0], [0], color=color_map[model], lw=4, label=model)
                      for model in model_types]
    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.02), ncol=min(4, len(model_types)),
              fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    return fig

# Create a comparison plot with all models on the same axes
def create_comparison_plots(fig_size=(16, 12)):
    """Create comparison plots with all models on the same axes."""
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create 4 panels: all probability, all surprisal, human vs LLMs, top models
    ax_prob = fig.add_subplot(gs[0, 0])
    ax_surp = fig.add_subplot(gs[0, 1])
    ax_human = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])
    
    # Set titles
    ax_prob.set_title('All Probability Models')
    ax_surp.set_title('All Surprisal Models')
    ax_human.set_title('Human Cloze Models')
    ax_top.set_title('Top LLM Models')
    
    # Set labels
    for ax in [ax_prob, ax_surp, ax_human, ax_top]:
        ax.set_ylabel('Reading Time (ms)')
    
    ax_prob.set_xlabel('Probability')
    ax_surp.set_xlabel('Surprisal')
    ax_human.set_xlabel('Predictor Value')
    ax_top.set_xlabel('Probability')
    
    # Plot all probability models
    for model_type in model_types:
        # Filter data
        model_smooth = smooth_data[smooth_data['model_type'] == model_type]
        
        # Get probability data
        prob_data = model_smooth[model_smooth['predictor_type'] == 'Probability']
        
        # Skip if no data
        if len(prob_data) == 0:
            continue
        
        # Plot on the probability panel
        ax_prob.plot(prob_data['x'], prob_data['y'],
                   color=color_map[model_type], linewidth=2,
                   label=model_type)
    
    # Plot all surprisal models
    for model_type in model_types:
        # Filter data
        model_smooth = smooth_data[smooth_data['model_type'] == model_type]
        
        # Get surprisal data
        surp_data = model_smooth[model_smooth['predictor_type'] == 'Surprisal']
        
        # Skip if no data
        if len(surp_data) == 0:
            continue
        
        # Plot on the surprisal panel
        ax_surp.plot(surp_data['x'], surp_data['y'],
                   color=color_map[model_type], linewidth=2,
                   label=model_type)
    
    # Plot human cloze models
    human_model = 'Human Cloze'
    if human_model in model_types:
        # Filter data
        human_smooth = smooth_data[smooth_data['model_type'] == human_model]
        
        # Plot probability
        prob_data = human_smooth[human_smooth['predictor_type'] == 'Probability']
        if len(prob_data) > 0:
            ax_human.plot(prob_data['x'], prob_data['y'],
                         color=color_map[human_model], linewidth=2.5,
                         label='Probability')
            ax_human.fill_between(prob_data['x'],
                                 prob_data['lower'],
                                 prob_data['upper'],
                                 color=color_map[human_model], alpha=0.2)
        
        # Plot surprisal
        surp_data = human_smooth[human_smooth['predictor_type'] == 'Surprisal']
        if len(surp_data) > 0:
            ax_human.plot(surp_data['x'], surp_data['y'],
                         color=color_map[human_model], linewidth=2.5,
                         linestyle='--', label='Surprisal')
            ax_human.fill_between(surp_data['x'],
                                 surp_data['lower'],
                                 surp_data['upper'],
                                 color=color_map[human_model], alpha=0.1)
    
    # Plot top LLM models (GPT-NeoX, LLaMA-2 and OLMO-2)
    top_models = [model for model in model_types
                  if any(name in model for name in ['NeoX', 'LLaMA', 'OLMO'])]
    
    for model_type in top_models:
        # Filter data
        model_smooth = smooth_data[smooth_data['model_type'] == model_type]
        
        # Get probability data
        prob_data = model_smooth[model_smooth['predictor_type'] == 'Probability']
        
        # Skip if no data
        if len(prob_data) == 0:
            continue
        
        # Plot on the top models panel
        ax_top.plot(prob_data['x'], prob_data['y'],
                   color=color_map[model_type], linewidth=2,
                   label=model_type)
        ax_top.fill_between(prob_data['x'],
                           prob_data['lower'],
                           prob_data['upper'],
                           color=color_map[model_type], alpha=0.1)
    
    # Add legends
    ax_prob.legend()
    ax_surp.legend()
    ax_human.legend()
    ax_top.legend()
    
    # Set the title
    fig.suptitle('Comparison of Predictability Models', fontsize=24, y=0.98)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

# Create the plots
fig1 = create_model_plots(fig_size=(20, 28), show_raw_data=True, alpha_raw=0.05)
fig1.savefig('gam_plots_from_r_with_raw_data.png', dpi=300, bbox_inches='tight')

fig2 = create_model_plots(fig_size=(20, 28), show_raw_data=False)
fig2.savefig('gam_plots_from_r_smooth_only.png', dpi=300, bbox_inches='tight')

fig3 = create_comparison_plots(fig_size=(16, 12))
fig3.savefig('gam_comparison_plots.png', dpi=300, bbox_inches='tight')

plt.show()

# Additional: create a publication-quality figure
def create_publication_figure(selected_models=None, fig_size=(10, 8)):
    """Create a publication-quality figure with selected models."""
    
    if selected_models is None:
        # Default to a few interesting models
        selected_models = ['Human Cloze', 'GPT-NeoX', 'LLaMA-2']
    
    # Create figure with 2x1 grid
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=True)
    
    # Set titles and labels
    axes[0].set_title('Probability Models')
    axes[1].set_title('Surprisal Models')
    
    axes[0].set_xlabel('Probability')
    axes[1].set_xlabel('Surprisal')
    
    axes[0].set_ylabel('Reading Time (ms)')
    
    # Plot selected models
    for model_type in selected_models:
        if model_type in model_types:
            # Filter data
            model_smooth = smooth_data[smooth_data['model_type'] == model_type]
            
            # Get probability data and plot
            prob_data = model_smooth[model_smooth['predictor_type'] == 'Probability']
            if len(prob_data) > 0:
                axes[0].plot(prob_data['x'], prob_data['y'],
                            color=color_map[model_type], linewidth=2.5,
                            label=model_type)
                axes[0].fill_between(prob_data['x'],
                                    prob_data['lower'],
                                    prob_data['upper'],
                                    color=color_map[model_type], alpha=0.2)
            
            # Get surprisal data and plot
            surp_data = model_smooth[model_smooth['predictor_type'] == 'Surprisal']
            if len(surp_data) > 0:
                axes[1].plot(surp_data['x'], surp_data['y'],
                            color=color_map[model_type], linewidth=2.5,
                            label=model_type)
                axes[1].fill_between(surp_data['x'],
                                    surp_data['lower'],
                                    surp_data['upper'],
                                    color=color_map[model_type], alpha=0.2)
    
    # Add legend
    axes[0].legend()
    
    # Set the title
    fig.suptitle('Comparing Human Cloze and LLM Predictions', fontsize=16)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

# Create publication-quality figure
fig4 = create_publication_figure(selected_models=['Human Cloze', 'GPT-NeoX', 'LLaMA-2'])
fig4.savefig('publication_quality_plot.png', dpi=600, bbox_inches='tight')

print("All plots have been saved to disk.")
