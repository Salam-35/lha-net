"""
Tools for analyzing and visualizing learned scale importance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter


class ScaleAnalyzer:
    """Analyze learned scale selection patterns"""

    def __init__(self, model, save_dir: str = "analysis/scales"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def extract_scale_probabilities(self) -> Dict[str, np.ndarray]:
        """Extract scale probabilities from all PMSA levels"""
        stats = self.model.get_scale_statistics()

        probs = {}
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                probs[key] = value.detach().cpu().numpy()

        return probs

    def plot_scale_distribution(
        self,
        epoch: Optional[int] = None,
        show: bool = False
    ):
        """Plot scale probability distribution across levels"""
        probs = self.extract_scale_probabilities()

        if not probs:
            print("No scale probabilities found (model may not use adaptive PMSA)")
            return

        num_levels = len(probs)
        fig, axes = plt.subplots(1, num_levels, figsize=(5*num_levels, 4))

        if num_levels == 1:
            axes = [axes]

        scale_bank = self.model.hierarchical_pmsa.pmsa_modules['level_0'].scale_bank

        for i, (level_key, level_probs) in enumerate(probs.items()):
            ax = axes[i]

            # Bar plot
            bars = ax.bar(range(len(scale_bank)), level_probs, alpha=0.7)

            # Color code by importance
            for j, (bar, prob) in enumerate(zip(bars, level_probs)):
                if prob > 0.15:  # Top scales
                    bar.set_color('red')
                elif prob > 0.08:
                    bar.set_color('orange')
                else:
                    bar.set_color('gray')

            ax.set_xticks(range(len(scale_bank)))
            ax.set_xticklabels([f'{s:.2f}' for s in scale_bank], rotation=45)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Probability')
            ax.set_title(f'{level_key.replace("_", " ").title()}')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(level_probs) * 1.2)

        title = f'Learned Scale Distribution'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        save_name = f'scale_distribution_epoch{epoch if epoch else "final"}.png'
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        print(f"Saved scale distribution to {self.save_dir / save_name}")

        if show:
            plt.show()
        plt.close()

    def get_selected_scales(self) -> Dict[str, List[float]]:
        """Get the top-K selected scales at each level"""
        selected = {}

        for i in range(len(self.model.hierarchical_pmsa.channels_list)):
            pmsa = self.model.hierarchical_pmsa.pmsa_modules[f'level_{i}']

            if hasattr(pmsa, 'scale_selector'):
                _, scales = pmsa.scale_selector.select_top_k_scales()
                selected[f'level_{i}'] = scales

        return selected

    def print_scale_summary(self):
        """Print human-readable summary of learned scales"""
        print("\n" + "="*60)
        print("LEARNED SCALE SUMMARY")
        print("="*60)

        probs = self.extract_scale_probabilities()
        selected = self.get_selected_scales()

        scale_bank = self.model.hierarchical_pmsa.pmsa_modules['level_0'].scale_bank

        for i, (level_key, level_probs) in enumerate(probs.items()):
            print(f"\n{level_key.upper()}:")
            print("-" * 40)

            # Show all scales with probabilities
            for scale, prob in zip(scale_bank, level_probs):
                marker = "★" if scale in selected[f'level_{i}'] else " "
                print(f"  {marker} Scale {scale:.2f}×: {prob:.4f} ({prob*100:.1f}%)")

            # Show selected scales
            print(f"\n  → Selected scales: {selected[f'level_{i}']}")

        print("\n" + "="*60)

    def compare_with_baseline(self, baseline_scales: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0]):
        """Compare learned scales with baseline configuration"""
        selected = self.get_selected_scales()

        print("\n" + "="*60)
        print("COMPARISON WITH BASELINE")
        print("="*60)
        print(f"\nBaseline scales: {baseline_scales}")

        for level_key, learned_scales in selected.items():
            print(f"\n{level_key.upper()}:")
            print(f"  Learned: {learned_scales}")

            # Calculate overlap
            overlap = set(learned_scales) & set(baseline_scales)
            print(f"  Overlap: {len(overlap)}/{len(baseline_scales)} scales")
            print(f"  New scales: {set(learned_scales) - set(baseline_scales)}")
            print(f"  Dropped scales: {set(baseline_scales) - set(learned_scales)}")

        print("="*60)

    def save_scale_history(self, epoch: int, history_dict: Dict):
        """Save scale probability history for training visualization"""
        probs = self.extract_scale_probabilities()

        for level_key, level_probs in probs.items():
            if level_key not in history_dict:
                history_dict[level_key] = []

            history_dict[level_key].append({
                'epoch': epoch,
                'probabilities': level_probs.copy()
            })

        return history_dict

    def plot_scale_evolution(self, history_dict: Dict, show: bool = False):
        """Plot how scale probabilities evolved during training"""
        num_levels = len(history_dict)

        if num_levels == 0:
            print("No scale history to plot")
            return

        fig, axes = plt.subplots(num_levels, 1, figsize=(12, 4*num_levels))

        if num_levels == 1:
            axes = [axes]

        scale_bank = self.model.hierarchical_pmsa.pmsa_modules['level_0'].scale_bank

        for i, (level_key, history) in enumerate(history_dict.items()):
            ax = axes[i]

            epochs = [h['epoch'] for h in history]

            # Plot probability evolution for each scale
            for scale_idx, scale in enumerate(scale_bank):
                probs_over_time = [h['probabilities'][scale_idx] for h in history]
                ax.plot(epochs, probs_over_time, marker='o', label=f'{scale:.2f}×', alpha=0.7)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Probability')
            ax.set_title(f'{level_key.replace("_", " ").title()} - Scale Evolution')
            ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5), ncol=1)
            ax.grid(alpha=0.3)

        plt.tight_layout()

        save_name = 'scale_evolution.png'
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        print(f"Saved scale evolution plot to {self.save_dir / save_name}")

        if show:
            plt.show()
        plt.close()


def analyze_scale_organ_correlation(
    model,
    dataloader,
    device: str = 'cuda',
    num_samples: int = 20
):
    """
    Analyze correlation between selected scales and organ presence.

    This helps understand if the model learns to activate certain scales
    when specific organs are present in the image.
    """
    model.eval()

    scale_activations = {f'level_{i}': [] for i in range(4)}
    organ_presence = []

    print("\nAnalyzing scale-organ correlation...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            labels = batch['label']

            # Get scale selection info
            output = model(images, return_scale_info=True)
            scale_info = output['scale_info']

            # Record which scales were selected at each level
            for level_key, info in scale_info.items():
                if 'selected_scales' in info:
                    scale_activations[level_key].append(info['selected_scales'])

            # Record which organs are present
            unique_organs = [torch.unique(l).tolist() for l in labels]
            organ_presence.append(unique_organs)

            print(f"  Processed sample {i+1}/{num_samples}")

    # Analyze patterns
    print("\n" + "="*60)
    print("SCALE-ORGAN CORRELATION ANALYSIS")
    print("="*60)

    # Count organ frequencies
    all_organs = []
    for organs_list in organ_presence:
        for organs in organs_list:
            all_organs.extend(organs)

    organ_counts = Counter(all_organs)

    print("\nOrgan frequency in samples:")
    organ_names = {
        1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gall_bladder',
        5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
        9: 'postcava', 10: 'pancreas', 11: 'right_adrenal', 12: 'left_adrenal',
        13: 'duodenum', 14: 'bladder', 15: 'prostate/uterus'
    }

    for organ_id, count in sorted(organ_counts.items()):
        if organ_id > 0:  # Skip background
            name = organ_names.get(organ_id, f'organ_{organ_id}')
            print(f"  {name}: {count}/{num_samples}")

    print("\n" + "="*60)

    return scale_activations, organ_presence
