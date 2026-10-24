# -*- coding: utf-8 -*-
"""
Is Stage 3 really worse on specificity, or is the fixed-threshold comparison
confounded by calibration?

For every arch/fold/stage this measures, on the SAME held-out positives and the
SAME 13 control images, the full threshold sweep:
  - recall on positives (IoU >= 0.5)
  - false positives on controls
Then compares stages at MATCHED RECALL rather than matched threshold.
"""
import os, io, csv, json, datetime, importlib.util
from pathlib import Path

BASE = r'D:\Ahmad Tarawneh\Amoeba detection\YOLO2\YOLO Compare'
MAIN = os.path.join(BASE, 'Main2_R5.py')
CSV = os.path.join(BASE, 'datasets', 'dataset_Amoeba_R5', 'meta.csv')
IMG = os.path.join(BASE, 'datasets', 'dataset_Amoeba_R5')
KF2 = os.path.join(BASE, 'Fixed_KFold_Results_R5', 'kfold_training')
CTRL = os.path.join(BASE, 'datasets', 'Negative examples')
OUT = os.path.join(BASE, 'Fixed_KFold_Results_R5', 'calibration_check_R3_grouped.csv')

ARCHS = ['yolov8s', 'yolov8n', 'yolo11n']
STAGES = [('stage1_heavy_aug', 'stage1'), ('stage2_moderate_aug', 'stage2'),
          ('stage3_fine_tune', 'stage3')]
TH = [round(0.05 * i, 2) for i in range(1, 19)]


def log(m):
    print('[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), m), flush=True)


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    i = (xb - xa) * (yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua if ua > 0 else 0.0


def main():
    import glob
    from sklearn.model_selection import StratifiedGroupKFold
    from ultralytics import YOLO

    spec = importlib.util.spec_from_file_location('m2', MAIN)
    m2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m2)
    tr = m2.FixedKFoldYOLO(CSV, IMG, r'C:\Windows\Temp\_cal_scratch')
    data = tr.load_and_prepare_data()
    images = list(data['image_name'].unique())
    strat = [tr.image_groups.get_group(i)['original_class'].mode().iloc[0] for i in images]
    import csv as _csv, re as _re, io as _io
    _gmap = {}
    with _io.open(r"D:\Ahmad Tarawneh\Amoeba detection\YOLO2\YOLO Compare\image_groups_R5.csv", encoding="utf-8-sig") as _fh:
        for _r in _csv.DictReader(_fh):
            _gmap[int(_r["image_number"])] = _r["group_id"]
    def _grp(_n):
        _m = _re.match(r"image(\d+)", str(_n).lower())
        return _gmap.get(int(_m.group(1)), "single_" + str(_n)) if _m else str(_n)
    _groups = [_grp(n) for n in images]
    log("independent units: %d from %d images" % (len(set(_groups)), len(images)))
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(images, strat, groups=_groups))

    ctrl = sorted(glob.glob(os.path.join(CTRL, '*.jpg')) +
                  glob.glob(os.path.join(CTRL, '*.jpeg')) +
                  glob.glob(os.path.join(CTRL, '*.png')))
    log('control images: %d' % len(ctrl))

    fh = io.open(OUT, 'w', encoding='utf-8', newline='')
    w = csv.writer(fh)
    w.writerow(['arch', 'fold', 'stage', 'threshold', 'tp', 'fn', 'recall', 'fp_control'])

    for arch in ARCHS:
        for fold, (_, va_idx) in enumerate(splits):
            va = [images[i] for i in va_idx]
            gt = {}
            for n in va:
                ann = tr.image_groups.get_group(n)
                gt[n] = [[float(r['bbox_x']), float(r['bbox_y']),
                          float(r['bbox_x']) + float(r['bbox_width']),
                          float(r['bbox_y']) + float(r['bbox_height'])]
                         for _, r in ann.iterrows()]
            for st_dir, tag in STAGES:
                wp = os.path.join(KF2, arch, 'fold_%d' % fold, st_dir, 'weights', 'best.pt')
                if not os.path.isfile(wp):
                    continue
                model = YOLO(wp)

                pos = []
                for n in va:
                    r = model.predict(os.path.join(IMG, n), imgsz=640, conf=0.05,
                                      device='0', verbose=False)[0]
                    pos.append((n, [(float(b.conf[0]), [float(x) for x in b.xyxy[0]])
                                    for b in r.boxes]))
                cf = []
                for p in ctrl:
                    r = model.predict(p, imgsz=640, conf=0.05, device='0', verbose=False)[0]
                    cf += [float(b.conf[0]) for b in r.boxes]

                for t in TH:
                    tp = fn = 0
                    for n, preds in pos:
                        boxes = [bb for c, bb in preds if c >= t]
                        used = set()
                        for g in gt[n]:
                            hit = False
                            for j, bb in enumerate(boxes):
                                if j in used:
                                    continue
                                if iou(g, bb) >= 0.5:
                                    used.add(j); hit = True; break
                            tp += 1 if hit else 0
                            fn += 0 if hit else 1
                    rec = tp / float(tp + fn) if (tp + fn) else 0.0
                    w.writerow([arch, fold, tag, t, tp, fn, '%.4f' % rec,
                                sum(1 for c in cf if c >= t)])
                fh.flush()
                log('  %-8s fold%d %-7s done' % (arch, fold, tag))
    fh.close()
    log('CALIBRATION CHECK COMPLETE -> %s' % OUT)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
