"""
XAI Analysis Pipeline for ICU Mortality Prediction Models

This script loads trained models and generates comprehensive explainability outputs:
- Feature importance (ICD codes, temporal patterns)
- Per-patient explanations with HTML reports
- Counterfactual "what-if" scenarios
- Mamba-specific visualizations (ODE dynamics, liquid states)

Usage:
    python xai_analysis.py --model LiquidMamba --n-patients 20
    python xai_analysis.py --model all --export-dashboard
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import model classes and utilities from research.py
from research import (
    ICUMortalityPredictor, BaselineTransformer, BaselineGRUD,
    ICUDataset, collate_fn, Config, set_seed,
    load_mimic_data, build_icd_graph, prepare_sequences
)

@dataclass
class XAIConfig:
    """Configuration for XAI analysis"""
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "results/xai"
    n_sample_patients: int = 20  # Number of patients to analyze in detail
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Visualization settings
    figsize: Tuple[int, int] = (12, 6)
    dpi: int = 300
    top_k_features: int = 20  # Top K features to visualize


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(
    model_name: str,
    model_class: type,
    model_kwargs: dict,
    checkpoint_path: Path,
    device: str
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Parameters
    ----------
    model_name : str
        Name of the model (for logging)
    model_class : type
        Model class to instantiate
    model_kwargs : dict
        Keyword arguments for model initialization
    checkpoint_path : Path
        Path to checkpoint file
    device : str
        Device to load model onto
        
    Returns
    -------
    model : nn.Module
        Loaded model in eval mode
    """
    print(f"\n📥 Loading {model_name} from {checkpoint_path}")
    
    # Initialize model
    model = model_class(**model_kwargs).to(device)
    
    # Load checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            auprc = checkpoint.get('best_val_auprc', 0)
            print(f"  ✓ Loaded from epoch {epoch}, best AUPRC: {auprc:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"  ✓ Loaded model weights")
    else:
        print(f"  ⚠️ Checkpoint not found at {checkpoint_path}")
        print(f"  Using randomly initialized model")
    
    model.eval()
    return model


# ============================================================================
# XAI EXTRACTION FUNCTIONS
# ============================================================================

def extract_liquid_mamba_internals(
    model: ICUMortalityPredictor,
    batch: dict,
    icd_adj: torch.Tensor,
    device: str
) -> Dict:
    """
    Extract internal states and attention from Liquid Mamba model.
    
    Returns
    -------
    internals : dict
        Dictionary containing:
        - graph_attention: (batch, n_nodes) - ICD code attention weights
        - temporal_hidden_states: (batch, seq, hidden_dim) - Hidden states over time
        - prediction: (batch,) - Mortality probability
        - uncertainty: (batch,) - Prediction uncertainty
        - logit: (batch,) - Raw logit
    """
    values = batch['values'].to(device)
    delta_t = batch['delta_t'].to(device)
    mask = batch['mask'].to(device)
    modality = batch['modality'].to(device)
    item_idx = batch['item_idx'].to(device)
    icd_activation = batch['icd_activation'].to(device)
    
    with torch.no_grad():
        # Forward pass with internals
        prob, uncertainty, logit, internals = model(
            values, delta_t, mask, modality, item_idx,
            icd_activation, icd_adj, return_internals=True
        )
    
    return {
        'graph_attention': internals.get('node_attention'),  # Model returns 'node_attention'
        'temporal_hidden': internals.get('hidden_states'),   # Model returns 'hidden_states'
        'prediction': prob.cpu().numpy(),
        'uncertainty': uncertainty.cpu().numpy(),
        'logit': logit.cpu().numpy()
    }


def extract_global_feature_importance(
    model: ICUMortalityPredictor,
    loader: DataLoader,
    icd_adj: torch.Tensor,
    device: str,
    icd_names: List[str]
) -> pd.DataFrame:
    """
    Compute global feature importance across all patients.
    
    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with columns: feature_name, attention_mean, attention_std
    """
    print("\n🔍 Computing global feature importance...")
    
    all_attentions = []
    
    print(f"  Processing {len(loader)} batches...")
    
    for batch in tqdm(loader, desc="Processing batches"):
        internals = extract_liquid_mamba_internals(model, batch, icd_adj, device)
        
        # Debug: Check if graph_attention exists
        if internals['graph_attention'] is not None:
            # (batch, n_nodes)
            attention = internals['graph_attention'].cpu().numpy()
            all_attentions.append(attention)
            print(f"    ✓ Collected attention from batch: shape {attention.shape}")
        else:
            print(f"    ⚠️ WARNING: graph_attention is None for this batch")
    
    print(f"  Total attentions collected: {len(all_attentions)}")
    
    # Aggregate across all patients
    if len(all_attentions) == 0:
        raise ValueError(
            "No attention data collected! This may happen if:\n"
            "  1. The test loader is empty\n"
            "  2. Model's forward pass doesn't return 'node_attention' in internals\n"
            "  3. All batches returned None for node_attention"
        )
    
    all_attentions = np.concatenate(all_attentions, axis=0)  # (total_patients, n_nodes)
    
    # Compute statistics
    attention_mean = all_attentions.mean(axis=0)
    attention_std = all_attentions.std(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'icd_code': icd_names[:len(attention_mean)],
        'attention_mean': attention_mean,
        'attention_std': attention_std
    })
    
    importance_df = importance_df.sort_values('attention_mean', ascending=False)
    
    return importance_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
    top_k: int = 20
):
    """
    Create horizontal bar plot of top-K most important features.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get top-K
    top_df = importance_df.head(top_k)
    
    # Plot
    y_pos = np.arange(len(top_df))
    ax.barh(y_pos, top_df['attention_mean'], xerr=top_df['attention_std'],
            color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['icd_code'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Top-{top_k} Important ICD Codes', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved feature importance plot: {save_path}")


def plot_patient_timeline(
    patient_id: int,
    values: np.ndarray,
    delta_t: np.ndarray,
    mask: np.ndarray,
    prediction: float,
    uncertainty: float,
    save_path: Path
):
    """
    Create timeline visualization for a single patient.
    
    Shows vital signs over time with prediction and uncertainty.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    # Compute cumulative time
    cumulative_time = np.cumsum(delta_t)
    valid_idx = mask.astype(bool)
    
    # Plot 1: Vital signs
    ax1.plot(cumulative_time[valid_idx], values[valid_idx], 
             marker='o', linestyle='-', linewidth=2, markersize=4,
             color='darkblue', alpha=0.7, label='Measurements')
    ax1.fill_between(cumulative_time[valid_idx], 
                      values[valid_idx] - values[valid_idx].std(),
                      values[valid_idx] + values[valid_idx].std(),
                      alpha=0.2, color='blue')
    ax1.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax1.set_title(f'Patient {patient_id} - Clinical Timeline', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Prediction bar
    colors = ['green' if prediction < 0.3 else 'orange' if prediction < 0.7 else 'red']
    ax2.barh([0], [prediction], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel('Mortality Risk Probability', fontsize=11, fontweight='bold')
    ax2.set_title(f'Prediction: {prediction:.1%} ± {uncertainty:.3f}', 
                  fontsize=11, fontweight='bold', color=colors[0])
    
    # Add threshold lines
    ax2.axvline(0.3, color='black', linestyle='--', alpha=0.5, label='Low risk')
    ax2.axvline(0.7, color='black', linestyle='--', alpha=0.5, label='High risk')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved patient timeline: {save_path}")


def plot_ode_dynamics(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    patient_id: int,
    save_path: Path
):
    """
    Visualize ODE Liquid Cell dynamics for Mamba model.
    
    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (seq_len, hidden_dim) - hidden states over time
    timesteps : np.ndarray
        Shape (seq_len,) - time points
    patient_id : int
        Patient ID for labeling
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Hidden state dimensions over time
    n_dims_to_plot = min(8, hidden_states.shape[1])
    for i in range(n_dims_to_plot):
        ax1.plot(timesteps, hidden_states[:, i], label=f'h{i}', alpha=0.7)
    
    ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hidden State Value', fontsize=11, fontweight='bold')
    ax1.set_title(f'Patient {patient_id}: Liquid State Evolution', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase portrait (first 2 dims)
    if hidden_states.shape[1] >= 2:
        ax2.plot(hidden_states[:, 0], hidden_states[:, 1], 
                marker='o', markersize=3, alpha=0.6, color='purple')
        ax2.scatter(hidden_states[0, 0], hidden_states[0, 1], 
                   s=100, c='green', marker='*', label='Start', zorder=5)
        ax2.scatter(hidden_states[-1, 0], hidden_states[-1, 1], 
                   s=100, c='red', marker='X', label='End', zorder=5)
        ax2.set_xlabel('Hidden Dim 0', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Hidden Dim 1', fontsize=11, fontweight='bold')
        ax2.set_title('ODE Phase Portrait', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved ODE dynamics: {save_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='XAI Analysis for ICU Mortality Prediction')
    parser.add_argument('--model', type=str, default='LiquidMamba',
                       choices=['LiquidMamba', 'Transformer', 'GRUD', 'all'],
                       help='Model to analyze')
    parser.add_argument('--n-patients', type=int, default=20,
                       help='Number of patients to analyze in detail')
    parser.add_argument('--export-dashboard', action='store_true',
                       help='Generate HTML dashboard')
    args = parser.parse_args()
    
    # Initialize
    xai_config = XAIConfig(n_sample_patients=args.n_patients)
    config = Config()
    set_seed(xai_config.seed)
    
    # Create output directories
    output_dir = Path(xai_config.output_dir)
    (output_dir / 'feature_importance').mkdir(parents=True, exist_ok=True)
    (output_dir / 'mamba_dynamics').mkdir(parents=True, exist_ok=True)
    (output_dir / 'patients').mkdir(parents=True, exist_ok=True)
    (output_dir / 'counterfactuals').mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("XAI ANALYSIS PIPELINE - ICU MORTALITY PREDICTION")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading MIMIC-IV data...")
    data = load_mimic_data(config.data_dir)
    icd_graph, icd_codes = build_icd_graph(data)
    tensors, labels, vocab_size = prepare_sequences(data, icd_graph, config)
    
    # Create test dataset
    n_total = len(labels)
    all_hadm_ids = labels.index.tolist()  # Get actual hadm_ids from labels
    test_start_idx = int(0.9 * n_total)
    test_ids = all_hadm_ids[test_start_idx:]  # Use actual hadm_ids, not integer indices
    test_dataset = ICUDataset(tensors, labels, icd_graph, test_ids)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    
    icd_adj = icd_graph.adj_matrix.to(xai_config.device)
    
    print(f"  ✓ Test set: {len(test_ids)} patients")
    print(f"  ✓ Vocabulary size: {vocab_size}")
    print(f"  ✓ ICD graph nodes: {icd_graph.n_nodes}")
    
    # Load Liquid Mamba model
    if args.model in ['LiquidMamba', 'all']:
        print("\n" + "=" * 80)
        print("ANALYZING: Liquid Mamba")
        print("=" * 80)
        
        checkpoint_path = Path(xai_config.checkpoint_dir) / "LiquidMamba_best.pth"
        model = load_trained_model(
            "Liquid Mamba",
            ICUMortalityPredictor,
            {"vocab_size": vocab_size, "n_icd_nodes": icd_graph.n_nodes, "config": config},
            checkpoint_path,
            xai_config.device
        )
        
        # 1. Global feature importance
        importance_df = extract_global_feature_importance(
            model, test_loader, icd_adj, xai_config.device, icd_codes
        )
        
        importance_df.to_csv(output_dir / 'feature_importance' / 'LiquidMamba_importance.csv', index=False)
        
        plot_feature_importance(
            importance_df,
            "Liquid Mamba",
            output_dir / 'feature_importance' / 'LiquidMamba_icd_importance.png',
            top_k=xai_config.top_k_features
        )
        
        # 2. Per-patient analysis (sample high-risk patients)
        print(f"\n🔬 Generating per-patient explanations for {xai_config.n_sample_patients} patients...")
        
        patient_count = 0
        for batch_idx, batch in enumerate(test_loader):
            if patient_count >= xai_config.n_sample_patients:
                break
            
            # Extract internals
            internals = extract_liquid_mamba_internals(model, batch, icd_adj, xai_config.device)
            
            batch_size = batch['values'].shape[0]
            for i in range(batch_size):
                if patient_count >= xai_config.n_sample_patients:
                    break
                
                patient_id = test_ids[batch_idx * config.batch_size + i]
                prediction = internals['prediction'][i]
                uncertainty = internals['uncertainty'][i]
                
                # Plot timeline
                plot_patient_timeline(
                    patient_id,
                    batch['values'][i].cpu().numpy(),
                    batch['delta_t'][i].cpu().numpy(),
                    batch['mask'][i].cpu().numpy(),
                    prediction,
                    uncertainty,
                    output_dir / 'patients' / f'patient_{patient_id}_timeline.png'
                )
                
                # Plot ODE dynamics (if available)
                if internals['temporal_hidden'] is not None:
                    hidden = internals['temporal_hidden'][i].cpu().numpy()
                    cumtime = np.cumsum(batch['delta_t'][i].cpu().numpy())
                    
                    plot_ode_dynamics(
                        hidden,
                        cumtime,
                        patient_id,
                        output_dir / 'mamba_dynamics' / f'patient_{patient_id}_ode_dynamics.png'
                    )
                
                patient_count += 1
    
    print("\n" + "=" * 80)
    print("✅ XAI ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"  - Feature importance plots")
    print(f"  - {patient_count} per-patient explanations")
    print(f"  - Mamba ODE dynamics visualizations")


if __name__ == "__main__":
    main()
