#!/usr/bin/env python3
"""Generate evaluation result comparison figures from hardcoded data.

Reads data from docs/eval_results_td_path.md and docs/eval_results_td_ego_dir.md
(hardcoded here) and produces PNG figures in docs/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style & color palette
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'AO': '#1f77b4',
    'TextCoT': '#ff7f0e',
    'VCoT l64': '#2ca02c',
    'MMCoT': '#d62728',
    'noEMA': '#1f77b4',
    'EMA': '#ff7f0e',
}
MARKERS = {'AO': 'o', 'TextCoT': 's', 'VCoT l64': '^', 'MMCoT': 'D'}
BASELINE_TD_PATH = 22.73  # SAT baseline, not used for td_path/dh_midpoint
# Baselines are not specified for td_path/dh_midpoint in the data, so we skip baseline lines there.


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ===================================================================
# Figure 1: td_path training curves (td_path subset)
# ===================================================================
def fig_td_path_training_curves():
    # AO
    ao_steps = [1500, 2400, 3000, 4500, 6000, 7500, 12000, 15000, 18000, 21000, 24000]
    ao_noema = [70.49, 74.06, 77.07, 78.76, 82.33, 80.26, 83.08, 82.33, 80.08, 81.20, 82.71]
    ao_ema   = [33.65, 34.77, 33.08, 38.91, 67.86, 75.00, 83.83, 85.90, 86.09, 85.15, 84.77]

    # TextCoT
    tc_noema_steps = [1500, 2000, 3000, 6000, 9000, 12000, 15000, 18000, 21000]
    tc_noema       = [62.22, 64.47, 65.04, 64.47, 66.04, 66.73, 62.22, 61.65, 60.90]
    tc_ema_steps   = [1500, 2000, 6000, 9000, 12000, 15000, 18000]
    tc_ema         = [31.39, 32.33, 49.44, 58.46, 59.21, 62.97, 63.16]

    # VCoT l64
    vc_noema_steps = [6000, 7000]
    vc_noema       = [40.23, 47.37]
    vc_ema_steps   = [6000, 7000]
    vc_ema         = [3.01, 40.04]

    # MMCoT
    mm_noema_steps = [6000, 7000]
    mm_noema       = [56.77, 49.44]
    mm_ema_steps   = [6000, 7000]
    mm_ema         = [43.98, 44.74]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # noEMA subplot
    ax = axes[0]
    ax.plot(ao_steps, ao_noema, color=COLORS['AO'], marker='o', label='AO', linewidth=2, markersize=5)
    ax.plot(tc_noema_steps, tc_noema, color=COLORS['TextCoT'], marker='s', label='TextCoT', linewidth=2, markersize=5)
    ax.plot(vc_noema_steps, vc_noema, color=COLORS['VCoT l64'], marker='^', label='VCoT l64', linewidth=2, markersize=6)
    ax.plot(mm_noema_steps, mm_noema, color=COLORS['MMCoT'], marker='D', label='MMCoT', linewidth=2, markersize=5)
    ax.set_title('noEMA', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=9)
    ax.set_ylim(25, 95)

    # EMA subplot
    ax = axes[1]
    ax.plot(ao_steps, ao_ema, color=COLORS['AO'], marker='o', label='AO', linewidth=2, markersize=5)
    ax.plot(tc_ema_steps, tc_ema, color=COLORS['TextCoT'], marker='s', label='TextCoT', linewidth=2, markersize=5)
    ax.plot(vc_ema_steps, vc_ema, color=COLORS['VCoT l64'], marker='^', label='VCoT l64', linewidth=2, markersize=6)
    ax.plot(mm_ema_steps, mm_ema, color=COLORS['MMCoT'], marker='D', label='MMCoT', linewidth=2, markersize=5)
    ax.set_title('EMA', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.legend(fontsize=9)
    ax.set_ylim(25, 95)

    fig.suptitle('td_path Accuracy vs Training Steps', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'td_path_training_curves.png')


# ===================================================================
# Figure 2: dh_midpoint training curves
# ===================================================================
def fig_dh_midpoint_training_curves():
    # AO
    ao_steps = [1500, 3000, 4500, 6000, 7500, 12000, 15000, 18000, 21000, 24000]
    ao_noema = [79.63, 74.07, 75.93, 80.86, 75.93, 75.31, 75.93, 69.14, 72.22, 70.37]
    ao_ema_steps = [1500, 3000, 4500, 6000, 7500, 12000, 15000, 18000, 21000, 24000]
    ao_ema   = [53.70, 56.17, 62.35, 80.86, 82.10, 80.86, 82.72, 80.86, 79.01, 77.16]

    # TextCoT
    tc_noema_steps = [1500, 2000, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000]
    tc_noema       = [72.22, 64.81, 68.52, 67.28, 64.81, 58.64, 53.70, 55.56, 50.62, 53.70]
    tc_ema_steps   = [1500, 2000, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000]
    tc_ema         = [56.79, 54.94, 43.83, 72.84, 69.75, 69.75, 64.81, 64.81, 62.35, 60.49]

    # VCoT l64
    vc_noema_steps = [6000, 7000, 9000, 12000, 15000]
    vc_noema       = [52.47, 49.38, 55.56, 58.02, 54.94]
    vc_ema_steps   = [6000, 7000, 9000, 12000, 15000]
    vc_ema         = [11.11, 52.47, 55.56, 62.35, 58.64]

    # MMCoT
    mm_noema_steps = [6000, 7000, 9000]
    mm_noema       = [61.11, 55.56, 59.88]
    mm_ema_steps   = [7000]
    mm_ema         = [67.90]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax = axes[0]
    ax.plot(ao_steps, ao_noema, color=COLORS['AO'], marker='o', label='AO', linewidth=2, markersize=5)
    ax.plot(tc_noema_steps, tc_noema, color=COLORS['TextCoT'], marker='s', label='TextCoT', linewidth=2, markersize=5)
    ax.plot(vc_noema_steps, vc_noema, color=COLORS['VCoT l64'], marker='^', label='VCoT l64', linewidth=2, markersize=6)
    ax.plot(mm_noema_steps, mm_noema, color=COLORS['MMCoT'], marker='D', label='MMCoT', linewidth=2, markersize=5)
    ax.set_title('noEMA', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=9)
    ax.set_ylim(30, 90)

    ax = axes[1]
    ax.plot(ao_ema_steps, ao_ema, color=COLORS['AO'], marker='o', label='AO', linewidth=2, markersize=5)
    ax.plot(tc_ema_steps, tc_ema, color=COLORS['TextCoT'], marker='s', label='TextCoT', linewidth=2, markersize=5)
    ax.plot(vc_ema_steps, vc_ema, color=COLORS['VCoT l64'], marker='^', label='VCoT l64', linewidth=2, markersize=6)
    ax.plot(mm_ema_steps, mm_ema, color=COLORS['MMCoT'], marker='D', label='MMCoT', linewidth=2, markersize=5)
    ax.set_title('EMA', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.legend(fontsize=9)
    ax.set_ylim(30, 90)

    fig.suptitle('dh_midpoint Accuracy vs Training Steps', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'dh_midpoint_training_curves.png')


# ===================================================================
# Figure 3: Cross-model best accuracy (grouped bars)
# ===================================================================
def fig_cross_model_best_accuracy():
    subsets = ['td_path', 'dh_midpoint', 'SAT_perspective']
    models = ['AO', 'TextCoT', 'VCoT l64', 'MMCoT']

    # Best values from cross-model comparison tables
    best_noema = {
        'AO':       [83.08, 80.86, 48.48],
        'TextCoT':  [66.73, 72.22, 56.06],
        'VCoT l64': [47.37, 58.02, 50.00],
        'MMCoT':    [56.77, 61.11, 31.82],
    }
    best_ema = {
        'AO':       [86.09, 82.72, 45.45],
        'TextCoT':  [63.16, 72.84, 51.52],
        'VCoT l64': [40.04, 62.35, 42.42],
        'MMCoT':    [44.74, 67.90, 57.58],
    }

    x = np.arange(len(subsets))
    n_models = len(models)
    bar_width = 0.15
    gap = 0.02  # gap between noEMA and EMA bars of same model

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * (2 * bar_width + gap)
        color = COLORS[model]
        # noEMA bar (solid)
        ax.bar(x + offset - bar_width / 2 - gap / 2, best_noema[model],
               bar_width, color=color, alpha=0.85, label=f'{model} noEMA',
               edgecolor='white', linewidth=0.5)
        # EMA bar (hatched)
        ax.bar(x + offset + bar_width / 2 + gap / 2, best_ema[model],
               bar_width, color=color, alpha=0.55, hatch='///',
               label=f'{model} EMA', edgecolor=color, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=11)
    ax.set_ylabel('Best Accuracy (%)', fontsize=11)
    ax.set_title('Cross-Model Best Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(0, 100)

    fig.tight_layout()
    save(fig, 'cross_model_best_accuracy.png')


# ===================================================================
# Figure 4: VCoT l64 EMA nothink rise-then-collapse
# ===================================================================
def fig_vcot_l64_nothink_collapse():
    steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 9000, 12000, 15000]
    acc   = [36.1, 37.8, 40.2, 46.2, 50.8, 59.2, 65.4, 63.0, 3.4, 0.0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, acc, color='#2ca02c', marker='o', linewidth=2.5, markersize=7,
            label='VCoT l64 EMA nothink (td_path)')

    # Annotate peak
    ax.annotate('Peak: 65.4%\n(s7000)', xy=(7000, 65.4), xytext=(8500, 75),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))
    # Annotate collapse
    ax.annotate('Collapse: 3.4%\n(s12000)', xy=(12000, 3.4), xytext=(12500, 25),
                fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='mistyrose', ec='red'))

    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('VCoT l64 EMA Nothink: Rise-then-Collapse on td_path', fontsize=13, fontweight='bold')
    ax.set_ylim(-5, 85)
    ax.legend(fontsize=10)

    save(fig, 'vcot_l64_nothink_collapse.png')


# ===================================================================
# Figure 5: AO td_ego_dir training curves
# ===================================================================
def fig_ao_ego_dir_training_curves():
    steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    pt2p_noema = [76.3, 79.0, 78.1, 77.5, 76.9, 78.1, 78.4, 75.1, 79.3, 80.2]
    pt2p_ema   = [53.8, 77.5, 79.0, 79.0, 79.3, 80.9, 79.6, 79.9, 79.0, 79.6]
    sv_noema   = [66.7, 71.2, 70.7, 68.2, 71.7, 70.2, 66.2, 69.7, 71.7, 72.2]
    sv_ema     = [59.6, 71.2, 69.2, 67.7, 72.7, 72.2, 72.2, 71.7, 73.7, 73.7]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax = axes[0]
    ax.plot(steps, pt2p_noema, color=COLORS['noEMA'], marker='o', label='noEMA', linewidth=2, markersize=5)
    ax.plot(steps, pt2p_ema, color=COLORS['EMA'], marker='s', label='EMA', linewidth=2, markersize=5)
    ax.set_title('PT2P (4-choice MCQ)', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=10)
    ax.set_ylim(45, 90)

    ax = axes[1]
    ax.plot(steps, sv_noema, color=COLORS['noEMA'], marker='o', label='noEMA', linewidth=2, markersize=5)
    ax.plot(steps, sv_ema, color=COLORS['EMA'], marker='s', label='EMA', linewidth=2, markersize=5)
    ax.set_title('SV (binary)', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.legend(fontsize=10)
    ax.set_ylim(45, 90)

    fig.suptitle('AO td_ego_dir: Accuracy vs Training Steps', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'ao_ego_dir_training_curves.png')


# ===================================================================
# Figure 6: VCoT l32 td_ego_dir eval comparison (bars)
# ===================================================================
def fig_vcot_l32_eval_comparison():
    # PT2P accuracy at s1k and s2k across eval modes
    labels = [
        'VCoT img-gen\nnoEMA', 'VCoT img-gen\nEMA',
        'Text think\nnoEMA', 'Text think\nEMA',
        'Nothink\nnoEMA', 'Nothink\nEMA',
    ]
    # PT2P values
    s1k_pt2p = [np.nan, 47.4, 71.7, 55.3, 72.0, 56.5]
    s2k_pt2p = [np.nan, np.nan, np.nan, 68.1, 71.7, 74.5]
    # SV values
    s1k_sv = [67.2, 55.6, 67.2, 57.6, 67.2, 56.6]
    s2k_sv = [55.1, 60.6, 68.2, 66.2, 68.7, 64.6]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    x = np.arange(len(labels))
    bar_w = 0.35

    # PT2P subplot
    ax = axes[0]
    bars1 = ax.bar(x - bar_w / 2, s1k_pt2p, bar_w, label='s1000', color='#5b9bd5', edgecolor='white')
    bars2 = ax.bar(x + bar_w / 2, s2k_pt2p, bar_w, label='s2000', color='#ed7d31', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('PT2P (4-choice MCQ)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 85)
    # Add value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.0f}',
                        ha='center', va='bottom', fontsize=7)

    # SV subplot
    ax = axes[1]
    bars1 = ax.bar(x - bar_w / 2, s1k_sv, bar_w, label='s1000', color='#5b9bd5', edgecolor='white')
    bars2 = ax.bar(x + bar_w / 2, s2k_sv, bar_w, label='s2000', color='#ed7d31', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('SV (binary)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 85)
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.0f}',
                        ha='center', va='bottom', fontsize=7)

    fig.suptitle('VCoT l32 td_ego_dir: Eval Setting Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'vcot_l32_eval_comparison.png')


# ===================================================================
# Figure 7: RealPathTracing overfitting pattern
# ===================================================================
def fig_realpt_training_curves():
    steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    # td_path
    tp_ema   = [39.1, 49.4, 47.7, 48.3, 48.9, 46.6, 45.4, 43.1]
    tp_noema = [46.6, 45.4, 44.3, 42.0, 43.1, 41.4, 44.8, 40.2]

    # td_path_arrow
    ta_ema   = [60.1, 74.1, 68.4, 63.9, 63.9, 62.7, 57.6, 56.3]
    ta_noema = [67.1, 66.5, 59.5, 58.2, 63.9, 56.3, 57.0, 51.9]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax = axes[0]
    ax.plot(steps, tp_noema, color=COLORS['noEMA'], marker='o', label='noEMA', linewidth=2, markersize=5)
    ax.plot(steps, tp_ema, color=COLORS['EMA'], marker='s', label='EMA', linewidth=2, markersize=5)
    ax.set_title('td_path (real indoor)', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=10)
    ax.set_ylim(30, 80)
    # Annotate EMA peak
    ax.annotate('EMA peak: 49.4%', xy=(2000, 49.4), xytext=(3500, 55),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='gray'))

    ax = axes[1]
    ax.plot(steps, ta_noema, color=COLORS['noEMA'], marker='o', label='noEMA', linewidth=2, markersize=5)
    ax.plot(steps, ta_ema, color=COLORS['EMA'], marker='s', label='EMA', linewidth=2, markersize=5)
    ax.set_title('td_path_arrow (real indoor)', fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.legend(fontsize=10)
    ax.set_ylim(30, 80)
    ax.annotate('EMA peak: 74.1%', xy=(2000, 74.1), xytext=(3500, 78),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='gray'))

    fig.suptitle('RealPathTracing: AO td_ego_dir Overfitting Pattern', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'realpt_training_curves.png')


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    print('Generating eval result figures...')
    fig_td_path_training_curves()
    fig_dh_midpoint_training_curves()
    fig_cross_model_best_accuracy()
    fig_vcot_l64_nothink_collapse()
    fig_ao_ego_dir_training_curves()
    fig_vcot_l32_eval_comparison()
    fig_realpt_training_curves()
    print('Done! All figures saved to docs/figures/')
