# -*- coding: utf-8 -*-
"""
make_qual_figures.py - qualitative detection figures, from the released data.

    python make_qual_figures.py --package <package> --results <results> --out figures/

Fig1   dataset samples (internal top row, control bottom row)
Fig7   three fold-0 held-out images, all three architectures
Fig8   three fold-1 held-out images, all three architectures
S1-S3  further qualitative comparisons
FigExt three external images, all three architectures

Green = ground truth, red = prediction. Each internal image is scored by the
fold model that did NOT train on it.
"""
import os, io, csv, json, argparse, re

ARCHS = ['yolov8s', 'yolov8n', 'yolo11n']
NAME = {'yolov8s': 'YOLOv8s', 'yolov8n': 'YOLOv8n', 'yolo11n': 'YOLO11n'}
GT, PR = (34, 160, 34), (220, 30, 30)


def load_gt(csv_path):
    d = {}
    for r in csv.DictReader(io.open(csv_path, encoding='utf-8-sig')):
        n = (r.get('image_name') or '').strip()
        if not n:
            continue
        x, y = float(r['bbox_x']), float(r['bbox_y'])
        w, h = float(r['bbox_width']), float(r['bbox_height'])
        d.setdefault(n, []).append([x, y, x + w, y + h])
    return d


def panel(path, gt, pred, confs, title, width=520):
    from PIL import Image, ImageDraw
    im = Image.open(path).convert('RGB')
    d = ImageDraw.Draw(im)
    for b in gt:
        d.rectangle(b, outline=GT, width=3)
    for b, c in zip(pred, confs):
        d.rectangle(b, outline=PR, width=2)
        d.text((b[0] + 2, max(0, b[1] - 12)), '%.2f' % c, fill=PR)
    w, h = im.size
    s = width / float(max(w, h))
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    strip = Image.new('RGB', (im.size[0], im.size[1] + 16), (255, 255, 255))
    strip.paste(im, (0, 16))
    ImageDraw.Draw(strip).text((3, 3), title, fill=(0, 0, 0))
    return strip


def sheet(panels, cols, path):
    from PIL import Image
    if not panels:
        return
    pw = max(p.size[0] for p in panels)
    ph = max(p.size[1] for p in panels)
    rows = (len(panels) + cols - 1) // cols
    c = Image.new('RGB', (cols * pw + (cols + 1) * 6, rows * ph + (rows + 1) * 6),
                  (255, 255, 255))
    for i, p in enumerate(panels):
        r, q = divmod(i, cols)
        c.paste(p, (6 + q * (pw + 6), 6 + r * (ph + 6)))
    c.save(path, dpi=(300, 300))
    print('  %s (%d panels)' % (os.path.basename(path), len(panels)))


def main():
    ap = argparse.ArgumentParser()
    B = r'D:\Ahmad Tarawneh\Amoeba detection\YOLO2\YOLO Compare'
    ap.add_argument('--package', default=os.path.join(B, 'package'))
    ap.add_argument('--results', default=os.path.join(B, 'Fixed_KFold_Results_R5'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--conf', type=float, default=0.25)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, 'figures')
    os.makedirs(out, exist_ok=True)

    from ultralytics import YOLO
    from PIL import Image

    DATA = os.path.join(a.package, 'data')
    IINT = os.path.join(DATA, 'internal', 'images')
    ICTL = os.path.join(DATA, 'control', 'images')
    IEXT = os.path.join(DATA, 'external', 'images')
    gt_int = load_gt(os.path.join(DATA, 'internal', 'annotations.csv'))
    gt_ext = load_gt(os.path.join(DATA, 'external', 'annotations.csv'))
    folds = json.load(io.open(os.path.join(a.results, 'true_folds.json'),
                              encoding='utf-8'))

    def wpath(arch, f):
        return os.path.join(a.results, 'kfold_training', arch, 'fold_%d' % f,
                            'stage3_fine_tune', 'weights', 'best.pt')

    def grid(fold, names, fname):
        panels = []
        for arch in ARCHS:
            m = YOLO(wpath(arch, fold))
            for n in names:
                p = os.path.join(IINT, n)
                if not os.path.isfile(p):
                    continue
                r = m.predict(p, imgsz=640, conf=a.conf, device='0', verbose=False)[0]
                pred = [[float(v) for v in b.xyxy[0]] for b in r.boxes]
                cf = [float(b.conf[0]) for b in r.boxes]
                panels.append(panel(p, gt_int.get(n, []), pred, cf,
                                    '%s  %s (fold %d held out)' % (NAME[arch], n, fold)))
        sheet(panels, 3, os.path.join(out, fname))

    # Fig 7 and Fig 8: three held-out images from folds 0 and 1
    grid(0, folds['0'][:3], 'Fig7.png')
    grid(1, folds['1'][:3], 'Fig8.png')
    # S1-S3: further held-out examples
    grid(2, folds['2'][:3], 'S1Fig.png')
    grid(3, folds['3'][:3], 'S2Fig.png')
    grid(4, folds['4'][:3], 'S3Fig.png')

    # external qualitative
    exs = sorted(os.listdir(IEXT),
                 key=lambda f: int(re.match(r'image(\d+)', f).group(1)))[:3]
    panels = []
    for arch in ARCHS:
        m = YOLO(wpath(arch, 0))
        for n in exs:
            p = os.path.join(IEXT, n)
            r = m.predict(p, imgsz=1280, conf=a.conf, device='0', verbose=False)[0]
            pred = [[float(v) for v in b.xyxy[0]] for b in r.boxes]
            cf = [float(b.conf[0]) for b in r.boxes]
            panels.append(panel(p, gt_ext.get(n, []), pred, cf,
                                '%s  external %s' % (NAME[arch], n)))
    sheet(panels, 3, os.path.join(out, 'FigExternal.png'))

    # Fig 1: dataset samples, internal over control
    panels = []
    for n in ['image1.jpeg', 'image2.jpeg', 'image88.jpeg']:
        p = os.path.join(IINT, n)
        if os.path.isfile(p):
            panels.append(panel(p, gt_int.get(n, []), [], [], 'main set  %s' % n))
    for n in sorted(os.listdir(ICTL),
                    key=lambda f: int(re.match(r'image(\d+)', f).group(1)))[:3]:
        panels.append(panel(os.path.join(ICTL, n), [], [], [], 'control set  %s' % n))
    sheet(panels, 3, os.path.join(out, 'Fig1.png'))


if __name__ == '__main__':
    main()
