# -*- coding: utf-8 -*-
"""
make_data_figures.py - regenerate every quantitative figure from the results.

    python make_data_figures.py --results <Fixed_KFold_Results_R5> --out figures/

Produces Fig3, Fig4, Fig5, Fig6, Fig9 and S4Fig at 300 dpi.
All values are read from the result files; nothing is hard-coded.
"""
import os, io, csv, json, argparse, statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCHS = ['yolov8s', 'yolov8n', 'yolo11n']
NAME = {'yolov8s': 'YOLOv8s', 'yolov8n': 'YOLOv8n', 'yolo11n': 'YOLO11n'}
COL = {'yolov8s': '#1f77b4', 'yolov8n': '#ff7f0e', 'yolo11n': '#2ca02c'}
STAGES = [('stage1_heavy_aug', 'Stage 1'), ('stage2_moderate_aug', 'Stage 2'),
          ('stage3_fine_tune', 'Stage 3')]
MET = [('precision', 'Precision'), ('recall', 'Recall'),
       ('f1', 'F1-score'), ('mAP50', 'mAP@50')]
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def folds(R, a):
    p = os.path.join(R, '%s_kfold_results.json' % a)
    d = json.load(io.open(p, encoding='utf-8'))
    return sorted(d['fold_results'], key=lambda x: x['fold'])


def stage_final(R, a, f, sd, key):
    p = os.path.join(R, 'kfold_training', a, 'fold_%d' % f, sd, 'results.csv')
    if not os.path.isfile(p):
        return None
    rows = list(csv.DictReader(io.open(p, encoding='utf-8', errors='replace')))
    if not rows:
        return None
    for k in rows[-1]:
        if k.strip() == key:
            try:
                return float(rows[-1][k])
            except Exception:
                return None
    return None


def fig3(R, out):
    """Per-fold metrics for each architecture, with deployment reference lines."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (key, lab) in zip(axes, MET):
        for a in ARCHS:
            v = [x[key] for x in folds(R, a)]
            ax.plot(range(5), v, 'o-', color=COL[a], label=NAME[a],
                    markersize=6, linewidth=1.6)
        if key in ('precision', 'recall'):
            ax.axhline(0.80, ls='--', c='grey', lw=1)
            ax.text(4.05, 0.80, ' 0.80', va='center', fontsize=7, color='grey')
        ax.set_title(lab)
        ax.set_xlabel('Fold')
        ax.set_xticks(range(5))
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel('Score')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'Fig3.png'), dpi=300)
    plt.close(fig)
    print('  Fig3.png')


def fig4(R, out):
    """Absolute performance after each training stage."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (key, lab) in zip(axes, [('metrics/precision(B)', 'Precision'),
                                     ('metrics/recall(B)', 'Recall'),
                                     ('metrics/mAP50(B)', 'mAP@50'),
                                     ('metrics/mAP50-95(B)', 'mAP@50-95')]):
        for a in ARCHS:
            means = []
            for sd, _ in STAGES:
                v = [stage_final(R, a, f, sd, key) for f in range(5)]
                v = [x for x in v if x is not None]
                means.append(st.mean(v) if v else float('nan'))
            ax.plot([1, 2, 3], means, 'o-', color=COL[a], label=NAME[a],
                    markersize=7, linewidth=1.8)
        ax.set_title(lab)
        ax.set_xlabel('Training stage')
        ax.set_xticks([1, 2, 3])
    axes[0].set_ylabel('Score (mean over 5 folds)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'Fig4.png'), dpi=300)
    plt.close(fig)
    print('  Fig4.png')


def fig5(R, out):
    """Change in performance contributed by each stage transition."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (key, lab) in zip(axes, [('metrics/mAP50(B)', 'mAP@50'),
                                     ('metrics/mAP50-95(B)', 'mAP@50-95')]):
        w = 0.26
        for i, a in enumerate(ARCHS):
            means = []
            for sd, _ in STAGES:
                v = [stage_final(R, a, f, sd, key) for f in range(5)]
                v = [x for x in v if x is not None]
                means.append(st.mean(v) if v else float('nan'))
            gains = [means[1] - means[0], means[2] - means[1]]
            ax.bar([x + (i - 1) * w for x in range(2)], gains, w,
                   color=COL[a], label=NAME[a])
        ax.axhline(0, c='black', lw=0.8)
        ax.set_xticks(range(2))
        ax.set_xticklabels(['Stage 1 \u2192 2', 'Stage 2 \u2192 3'])
        ax.set_title(lab)
    axes[0].set_ylabel('Change in score')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'Fig5.png'), dpi=300)
    plt.close(fig)
    print('  Fig5.png')


def fig6(R, out):
    """Each metric as a function of the confidence threshold, per fold."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for r, a in enumerate(ARCHS):
        fr = folds(R, a)
        for c, (key, lab) in enumerate([('precision', 'Precision'),
                                        ('recall', 'Recall'),
                                        ('mAP50', 'mAP@50')]):
            ax = axes[r][c]
            for x in fr:
                to = x.get('threshold_optimization', {})
                series = None
                for v in to.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict) \
                            and 'conf_threshold' in v[0]:
                        series = v
                        break
                if not series:
                    continue
                series = sorted(series, key=lambda z: z['conf_threshold'])
                ax.plot([z['conf_threshold'] for z in series],
                        [z.get(key, float('nan')) for z in series],
                        '-o', markersize=3, linewidth=1.1,
                        label='Fold %d' % x['fold'])
            if r == 0:
                ax.set_title(lab)
            if c == 0:
                ax.set_ylabel('%s\nScore' % NAME[a])
            if r == 2:
                ax.set_xlabel('Confidence threshold')
    axes[0][2].legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'Fig6.png'), dpi=300)
    plt.close(fig)
    print('  Fig6.png')


def _ctrl(R):
    for n in ('control_fp_counts_R2.csv', 'control_fp_counts_R4.csv',
              'control_fp_counts_R5.csv'):
        p = os.path.join(R, n)
        if os.path.isfile(p):
            return list(csv.DictReader(io.open(p, encoding='utf-8')))
    return []


def fig9(R, out):
    """False positives on the 13 negative images, Stage-3 models."""
    rows = _ctrl(R)
    TH = sorted(set(round(float(r['threshold']), 2) for r in rows))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for a in ARCHS:
        tot = [sum(int(r['false_positives']) for r in rows
                   if r['arch'] == a and r['stage'] == 'stage3'
                   and abs(float(r['threshold']) - t) < 1e-9) for t in TH]
        ax.plot(TH, tot, 'o-', color=COL[a], label=NAME[a], markersize=6)
    ax.set_xlabel('Confidence threshold')
    ax.set_ylabel('Total false positives (5 folds)')
    ax.set_title('False positives on the 13 negative images (Stage 3)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'Fig9.png'), dpi=300)
    plt.close(fig)
    print('  Fig9.png')


def s4(R, out):
    """False positives by threshold for all three stages."""
    rows = _ctrl(R)
    TH = sorted(set(round(float(r['threshold']), 2) for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (sd, lab) in zip(axes, [('stage1', 'Stage 1'), ('stage2', 'Stage 2'),
                                    ('stage3', 'Stage 3')]):
        for a in ARCHS:
            tot = [sum(int(r['false_positives']) for r in rows
                       if r['arch'] == a and r['stage'] == sd
                       and abs(float(r['threshold']) - t) < 1e-9) for t in TH]
            ax.plot(TH, tot, 'o-', color=COL[a], label=NAME[a], markersize=5)
        ax.set_title(lab)
        ax.set_xlabel('Confidence threshold')
    axes[0].set_ylabel('Total false positives (5 folds)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'S4Fig.png'), dpi=300)
    plt.close(fig)
    print('  S4Fig.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=(r'D:\Ahmad Tarawneh\Amoeba detection'
                                          r'\YOLO2\YOLO Compare\Fixed_KFold_Results_R5'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, 'figures')
    os.makedirs(out, exist_ok=True)
    print('figures -> %s' % out)
    fig3(a.results, out)
    fig4(a.results, out)
    fig5(a.results, out)
    try:
        fig6(a.results, out)
    except Exception as e:
        print('  Fig6 skipped: %s' % str(e)[:80])
    fig9(a.results, out)
    s4(a.results, out)


if __name__ == '__main__':
    main()
