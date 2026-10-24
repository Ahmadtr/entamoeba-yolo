# -*- coding: utf-8 -*-
"""R2 post-training evaluation: external validation (Table 5) + control-set FPs (Table 4)."""
import os, io, csv, glob, datetime

from ultralytics import YOLO

BASE = r'D:\Ahmad Tarawneh\Amoeba detection\YOLO2\YOLO Compare'
KF2 = os.path.join(BASE, 'Fixed_KFold_Results_R5', 'kfold_training')
OUT = os.path.join(BASE, 'Fixed_KFold_Results_R5')
EXTYAML = os.path.join(BASE, 'Fixed_KFold_Results', 'kfold_training', 'extval', 'extval.yaml')
CTRL = os.path.join(BASE, 'datasets', 'Negative examples')

ARCHS = ['yolov8s', 'yolov8n', 'yolo11n']
STAGES = [('stage1_heavy_aug', 'stage1'), ('stage2_moderate_aug', 'stage2'),
          ('stage3_fine_tune', 'stage3')]
THRESH = [round(0.1 * i, 1) for i in range(1, 10)]


def wpath(arch, fold, st_dir):
    return os.path.join(KF2, arch, 'fold_%d' % fold, st_dir, 'weights', 'best.pt')


def log(m):
    print('[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), m), flush=True)


# ---------------- 1. EXTERNAL VALIDATION (stage 3 only, two resolutions) ----------------
log('EXTERNAL VALIDATION starting')
p = os.path.join(OUT, 'extval_results_R2.csv')
fh = io.open(p, 'w', encoding='utf-8', newline='')
w = csv.writer(fh)
w.writerow(['arch', 'fold', 'imgsz', 'precision', 'recall', 'f1', 'map50', 'map50_95'])
for arch in ARCHS:
    for fold in range(5):
        wp = wpath(arch, fold, 'stage3_fine_tune')
        if not os.path.isfile(wp):
            log('  missing %s fold%d' % (arch, fold))
            continue
        model = YOLO(wp)
        for size in (1280, 640):
            res = model.val(data=EXTYAML, imgsz=size, batch=4, device='0',
                            workers=0, split='val', plots=False, verbose=False)
            b = res.box
            pr, rc = float(b.mp), float(b.mr)
            f1 = 2 * pr * rc / (pr + rc + 1e-12)
            w.writerow([arch, fold, size, '%.4f' % pr, '%.4f' % rc, '%.4f' % f1,
                        '%.4f' % float(b.map50), '%.4f' % float(b.map)])
            fh.flush()
            log('  %-8s fold%d imgsz%-5d mAP50=%.4f' % (arch, fold, size, b.map50))
fh.close()
log('EXTERNAL VALIDATION done -> %s' % p)

# ---------------- 2. CONTROL SET FALSE POSITIVES (all 45 models) ----------------
log('CONTROL SET starting')
imgs = sorted(glob.glob(os.path.join(CTRL, '*.jpg')) +
              glob.glob(os.path.join(CTRL, '*.jpeg')) +
              glob.glob(os.path.join(CTRL, '*.png')))
log('  control images: %d' % len(imgs))

p2 = os.path.join(OUT, 'control_fp_counts_R2.csv')
fh = io.open(p2, 'w', encoding='utf-8', newline='')
w = csv.writer(fh)
w.writerow(['arch', 'fold', 'stage', 'imgsz', 'n_images', 'threshold', 'false_positives'])
for arch in ARCHS:
    for fold in range(5):
        for st_dir, tag in STAGES:
            wp = wpath(arch, fold, st_dir)
            if not os.path.isfile(wp):
                continue
            model = YOLO(wp)
            confs = []
            for im in imgs:
                r = model.predict(im, imgsz=640, conf=0.05, device='0', verbose=False)[0]
                confs += [float(bb.conf[0]) for bb in r.boxes]
            for t in THRESH:
                w.writerow([arch, fold, tag, 640, len(imgs), t,
                            sum(1 for c in confs if c >= t)])
            fh.flush()
            log('  %-8s fold%d %-7s done' % (arch, fold, tag))
fh.close()
log('CONTROL SET done -> %s' % p2)
log('ALL EVALUATION COMPLETE')
