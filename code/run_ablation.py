# -*- coding: utf-8 -*-
"""
Single-stage ablation at matched epoch budget (Reviewer #1, required analysis 1).

Epoch budgets are read from budgets.json, which is frozen immediately after
training. They are never recomputed here, so nothing downstream can inflate them.

Three constant-augmentation variants (heavy / moderate / light) x 3 architectures
x 5 folds. Early stopping disabled so each run uses its full matched budget.
Folds come from the same StratifiedGroupKFold(seed=42) as the main run,
so patient/slide grouping is identical in both arms.
"""
import os, io, csv, json, re, datetime, importlib.util
from pathlib import Path

BASE = r'D:\Ahmad Tarawneh\Amoeba detection\YOLO2\YOLO Compare'
MAIN = os.path.join(BASE, 'Main2_R5.py')
CSV = os.path.join(BASE, 'datasets', 'dataset_Amoeba_R5', 'meta.csv')
IMG = os.path.join(BASE, 'datasets', 'dataset_Amoeba_R5')
R2 = os.path.join(BASE, 'Fixed_KFold_Results_R5')
OUT = os.path.join(R2, 'ablation_single_stage')
BUDGETS = os.path.join(R2, 'budgets.json')

ARCHS = ['yolov8s.pt', 'yolov8n.pt', 'yolo11n.pt']
VARIANTS = [(2, 'constant_moderate'), (1, 'constant_heavy'), (3, 'constant_light')]


def log(m):
    print('[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), m), flush=True)


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    from ultralytics import YOLO

    budgets = json.load(io.open(BUDGETS, encoding='utf-8'))
    log('loaded frozen budgets: %s' % budgets)

    spec = importlib.util.spec_from_file_location('m2', MAIN)
    m2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m2)

    trainer = m2.FixedKFoldYOLO(CSV, IMG, OUT)
    data = trainer.load_and_prepare_data()
    images = list(data['image_name'].unique())
    strat = [trainer.image_groups.get_group(i)['original_class'].mode().iloc[0] for i in images]
    # --- authoritative fold membership, as the training pipeline used it ---
    import json as _json, io as _io, os as _os
    _tf = _os.path.join(R4 if "R4" in globals() else r"D:\Ahmad Tarawneh\Amoeba detection\YOLO2\YOLO Compare\Fixed_KFold_Results_R5", "true_folds.json")
    if not _os.path.isfile(_tf):
        raise SystemExit("ABORT: true_folds.json not found at %s" % _tf)
    _folds = _json.load(_io.open(_tf, encoding="utf-8"))
    _imgset = set(images)
    splits = []
    for _f in range(5):
        _va = [n for n in _folds[str(_f)] if n in _imgset]
        if len(_va) != len(_folds[str(_f)]):
            raise SystemExit("ABORT: fold %d references unknown images" % _f)
        _vaset = set(_va)
        _tr_idx = [i for i, n in enumerate(images) if n not in _vaset]
        _va_idx = [i for i, n in enumerate(images) if n in _vaset]
        splits.append((_tr_idx, _va_idx))
    log("fold sizes from true_folds.json: %s" % [len(v) for _, v in splits])
    _allv = [n for _f in range(5) for n in _folds[str(_f)]]
    if len(_allv) != len(set(_allv)) or len(_allv) != len(images):
        raise SystemExit("ABORT: true_folds.json is not a clean partition")
    log('folds: %s' % [len(v) for _, v in splits])

    os.makedirs(OUT, exist_ok=True)
    rp = os.path.join(OUT, 'ablation_results.csv')
    fh = io.open(rp, 'w', encoding='utf-8', newline='')
    w = csv.writer(fh)
    w.writerow(['variant', 'arch', 'fold', 'epochs_budget',
                'precision', 'recall', 'f1',
                'mAP50', 'mAP75', 'mAP50_95',
                'fitness', 'n_images'])

    for stage, vname in VARIANTS:
        log('========== VARIANT: %s ==========' % vname)
        for arch in ARCHS:
            short = arch.replace('.pt', '')
            for fold, (tr_idx, va_idx) in enumerate(splits):
                budget = budgets.get('%s_fold%d' % (short, fold), 0)
                if budget <= 0:
                    log('  no budget for %s fold%d, skipping' % (short, fold))
                    continue
                tr = [images[i] for i in tr_idx]
                va = [images[i] for i in va_idx]

                # per-architecture directory, wiped each time: the three
                # architectures must NEVER share a dataset folder, or one
                # run's val images become the next run's training images
                ds = Path(os.path.join(OUT, vname, 'data', short, 'fold_%d' % fold))
                if ds.exists():
                    import shutil as _sh
                    _sh.rmtree(str(ds))
                ds.mkdir(parents=True, exist_ok=True)
                yaml_path = trainer.create_progressive_dataset(tr, va, ds, stage)

                # integrity gate: abort rather than train on contaminated data
                def _stems(_d):
                    if not os.path.isdir(_d):
                        return set()
                    return set(re.sub(r'_s\d_aug\d+$', '',
                                      os.path.splitext(_f)[0]).lower()
                               for _f in os.listdir(_d))
                _tr = _stems(os.path.join(str(ds), 'images', 'train'))
                _va = _stems(os.path.join(str(ds), 'images', 'val'))
                _ov = _tr & _va
                _expect = set(os.path.splitext(x)[0].lower() for x in va)
                if _ov:
                    raise SystemExit('ABORT: train/val overlap in %s: %s'
                                     % (ds, sorted(_ov)[:6]))
                if _va != _expect:
                    raise SystemExit('ABORT: val set != fold %d held-out set (%d vs %d)'
                                     % (fold, len(_va), len(_expect)))
                log('    dataset ok: train %d, val %d, no overlap' % (len(_tr), len(_va)))

                cfg = trainer.get_progressive_config(arch, stage, fold)
                cfg['data'] = str(yaml_path)
                cfg['epochs'] = budget
                cfg['patience'] = 0
                cfg['workers'] = 0
                cfg['project'] = os.path.join(OUT, vname, short, 'fold_%d' % fold)
                cfg['name'] = 'single_stage'
                cfg['exist_ok'] = True

                log('  %s %s fold%d : %d epochs' % (vname, short, fold, budget))
                YOLO(arch).train(**cfg)

                best = os.path.join(cfg['project'], 'single_stage', 'weights', 'best.pt')
                if not os.path.isfile(best):
                    log('    no weights produced')
                    continue
                vr = YOLO(best).val(data=str(yaml_path), conf=0.25, workers=0,
                                    verbose=False, save_json=True, plots=False)
                b = vr.box

                def g(obj, *names):
                    for nm in names:
                        v = getattr(obj, nm, None)
                        if v is None:
                            continue
                        try:
                            return float(v)
                        except Exception:
                            try:
                                return float(sum(v) / len(v))
                            except Exception:
                                pass
                    return float('nan')

                p = g(b, 'mp')
                r = g(b, 'mr')
                m50 = g(b, 'map50')
                m75 = g(b, 'map75')
                m95 = g(b, 'map')
                f1 = 2 * p * r / (p + r) if (p + r) else float('nan')
                fit = g(vr, 'fitness')
                w.writerow([vname, short, fold, budget,
                            '%.4f' % p, '%.4f' % r, '%.4f' % f1,
                            '%.4f' % m50, '%.4f' % m75, '%.4f' % m95,
                            '%.4f' % fit, len(va)])
                fh.flush()
                log('    mAP50=%.4f  mAP75=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f  F1=%.4f'
                    % (m50, m75, m95, p, r, f1))

                # keep the full per-stage curve too, for any later question
                try:
                    import shutil as _sh2
                    rc = os.path.join(cfg['project'], 'single_stage', 'results.csv')
                    if os.path.isfile(rc):
                        keep = os.path.join(OUT, 'curves')
                        os.makedirs(keep, exist_ok=True)
                        _sh2.copy2(rc, os.path.join(
                            keep, '%s_%s_fold%d_results.csv' % (vname, short, fold)))
                except Exception as _e:
                    log('    (curve copy failed: %s)' % _e)

    fh.close()
    log('ABLATION COMPLETE -> %s' % rp)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
