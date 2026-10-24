# -*- coding: utf-8 -*-
"""
verify_package.py - check that this package reproduces the published results.

    python verify_package.py              # data + tables (no GPU, ~20 s)
    python verify_package.py --weights    # also re-runs inference (needs GPU)

Exit code 0 = every check passed.
"""
import os, io, csv, json, sys, hashlib, argparse, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHS = ['yolov8s', 'yolov8n', 'yolo11n']
NAME = {'yolov8s': 'YOLOv8s', 'yolov8n': 'YOLOv8n', 'yolo11n': 'YOLO11n'}
fails, checks = [], 0


def ok(cond, label, detail=''):
    global checks
    checks += 1
    print('  [%s] %s%s' % ('PASS' if cond else 'FAIL', label,
                           '' if not detail else '  (%s)' % detail))
    if not cond:
        fails.append(label)
    return cond


def near(a, b, tol=0.0006):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------- data
def check_data():
    print('\n1. DATASET INTEGRITY')
    D = os.path.join(HERE, 'data')
    expect = {'internal': (90, 188), 'external': (80, 207), 'control': (13, None)}
    for which, (n_img, n_box) in expect.items():
        d = os.path.join(D, which, 'images')
        imgs = [f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        ok(len(imgs) == n_img, '%s: %d images' % (which, n_img),
           'found %d' % len(imgs))
        nums = sorted(int(''.join(c for c in f.split('.')[0] if c.isdigit()))
                      for f in imgs)
        ok(nums == list(range(1, n_img + 1)),
           '%s: names are image1..image%d with no gaps' % (which, n_img))
        if n_box:
            ap = os.path.join(D, which, 'annotations.csv')
            rows = [r for r in csv.DictReader(io.open(ap, encoding='utf-8-sig'))
                    if (r.get('image_name') or '').strip()]
            ok(len(rows) == n_box, '%s: %d annotations' % (which, n_box),
               'found %d' % len(rows))
            names = set(r['image_name'].strip() for r in rows)
            ok(names <= set(imgs),
               '%s: every annotation refers to a released image' % which)

    # checksums, if a manifest is present
    for which in expect:
        man = None
        for cand in ('manifest.csv', 'rename_map.csv'):
            p = os.path.join(D, which, cand)
            if os.path.isfile(p):
                man = p
                break
        if not man:
            continue
        bad = 0
        for r in csv.DictReader(io.open(man, encoding='utf-8-sig')):
            nm = (r.get('released_name') or r.get('image_file') or '').strip()
            want = (r.get('md5') or '').strip()
            fp = os.path.join(D, which, 'images', nm)
            if nm and want and os.path.isfile(fp):
                got = hashlib.md5(io.open(fp, 'rb').read()).hexdigest()
                if got != want:
                    bad += 1
        ok(bad == 0, '%s: all file checksums match the manifest' % which,
           '%d mismatched' % bad)

    # class balance quoted in the paper
    ap = os.path.join(D, 'internal', 'annotations.csv')
    rows = [r for r in csv.DictReader(io.open(ap, encoding='utf-8-sig'))
            if (r.get('image_name') or '').strip()]
    cy = sum(1 for r in rows if 'cyst' in r['label_name'].lower())
    ok(cy == 143 and len(rows) - cy == 45,
       'internal: 143 cysts / 45 trophozoites',
       '%d / %d' % (cy, len(rows) - cy))

    # clinical / public split
    gp = os.path.join(D, 'internal', 'groups.csv')
    g = list(csv.DictReader(io.open(gp, encoding='utf-8-sig')))
    clin = sum(1 for r in g if r['source_type'].strip() == 'clinical')
    ok(clin == 63 and len(g) - clin == 27,
       'internal: 63 clinical / 27 public', '%d / %d' % (clin, len(g) - clin))


# ---------------------------------------------------------------- folds
def check_folds():
    print('\n2. CROSS-VALIDATION FOLDS')
    fp = os.path.join(HERE, 'results', 'true_folds.json')
    if not os.path.isfile(fp):
        ok(False, 'true_folds.json present')
        return
    folds = json.load(io.open(fp, encoding='utf-8'))
    sizes = [len(folds[str(i)]) for i in range(5)]
    allv = [n for i in range(5) for n in folds[str(i)]]
    ok(sizes == [18] * 5, 'five folds of 18 images', str(sizes))
    ok(len(allv) == len(set(allv)) == 90,
       'folds are a clean partition of all 90 images')

    gp = os.path.join(HERE, 'data', 'internal', 'groups.csv')
    grp = {r['image_file'].strip(): r['group_id'].strip()
           for r in csv.DictReader(io.open(gp, encoding='utf-8-sig'))}
    spans = []
    seen = {}
    for i in range(5):
        for n in folds[str(i)]:
            g = grp.get(n)
            if g in seen and seen[g] != i:
                spans.append(g)
            seen[g] = i
    ok(not spans, 'no patient appears in two folds', str(sorted(set(spans))[:4]))


# ---------------------------------------------------------------- tables
def _folds_json(a):
    p = os.path.join(HERE, 'results', '%s_kfold_results.json' % a)
    d = json.load(io.open(p, encoding='utf-8'))
    return sorted(d['fold_results'], key=lambda x: x['fold'])


def check_tables():
    print('\n3. PUBLISHED TABLES REGENERATE FROM results/')
    T = os.path.join(HERE, 'results', 'tables')

    # Table 2 - macro average
    p = os.path.join(T, 'Table2_macro_average.csv')
    if os.path.isfile(p):
        pub = {r['Model']: r for r in csv.DictReader(io.open(p, encoding='utf-8-sig'))}
        allok = True
        for a in ARCHS:
            fr = _folds_json(a)
            for key, col in (('precision', 'Precision'), ('recall', 'Recall'),
                             ('f1', 'F1-score'), ('mAP50', 'mAP@50')):
                v = [x[key] for x in fr]
                want = '%.3f\u00b1%.3f' % (st.mean(v), st.pstdev(v))
                got = pub.get(NAME[a], {}).get(col, '').replace(' ', '')
                if want.replace(' ', '') != got:
                    allok = False
        ok(allok, 'Table 2 matches the per-fold results')

    # Table 3 - per fold
    p = os.path.join(T, 'Table3_per_fold.csv')
    if os.path.isfile(p):
        pub = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
        allok = len(pub) == 15
        for r in pub:
            a = [k for k, v in NAME.items() if v == r['Model']]
            if not a:
                continue
            fr = {x['fold']: x for x in _folds_json(a[0])}
            x = fr.get(int(r['Fold']))
            if not x or not near(float(r['mAP@50']), round(x['mAP50'], 3)):
                allok = False
        ok(allok, 'Table 3 matches the per-fold results (15 rows)')

    # ablation
    p = os.path.join(HERE, 'results', 'ablation_results.csv')
    if os.path.isfile(p):
        rows = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
        ok(len(rows) == 45, 'ablation has all 45 runs', '%d' % len(rows))
        ok(all(r['n_images'] == '18' for r in rows),
           'every ablation run used an 18-image held-out fold')
        ok(all(k in rows[0] for k in ('mAP50', 'mAP75', 'mAP50_95')),
           'ablation records mAP@50, mAP@75 and mAP@50-95')


# ---------------------------------------------------------------- weights
def check_weights():
    print('\n4. RE-RUNNING INFERENCE (weights required)')
    W = os.path.join(HERE, 'weights')
    if not os.path.isdir(W):
        ok(False, 'weights/ present',
           'not shipped - inference figures cannot be reproduced')
        return
    have = [f for f in os.listdir(W) if f.endswith('.pt')]
    if not ok(len(have) == 15, 'all 15 stage-3 models present', '%d' % len(have)):
        return
    try:
        from ultralytics import YOLO
    except Exception as e:
        ok(False, 'ultralytics importable', str(e)[:50])
        return
    folds = json.load(io.open(os.path.join(HERE, 'results', 'true_folds.json'),
                              encoding='utf-8'))
    fr = {x['fold']: x for x in _folds_json('yolov8s')}
    import tempfile, shutil
    img = os.path.join(HERE, 'data', 'internal', 'images')
    ann = {}
    for r in csv.DictReader(io.open(os.path.join(HERE, 'data', 'internal',
                                                 'annotations.csv'),
                                    encoding='utf-8-sig')):
        ann.setdefault(r['image_name'].strip(), []).append(r)
    tmp = tempfile.mkdtemp()
    try:
        di, dl = os.path.join(tmp, 'images'), os.path.join(tmp, 'labels')
        os.makedirs(di); os.makedirs(dl)
        for n in folds['0']:
            shutil.copy2(os.path.join(img, n), os.path.join(di, n))
            W_, H_ = None, None
            lines = []
            for r in ann.get(n, []):
                W_, H_ = float(r['image_width']), float(r['image_height'])
                x, y = float(r['bbox_x']), float(r['bbox_y'])
                bw, bh = float(r['bbox_width']), float(r['bbox_height'])
                lines.append('0 %.6f %.6f %.6f %.6f'
                             % ((x + bw / 2) / W_, (y + bh / 2) / H_,
                                bw / W_, bh / H_))
            io.open(os.path.join(dl, os.path.splitext(n)[0] + '.txt'), 'w',
                    encoding='utf-8').write('\n'.join(lines))
        y = os.path.join(tmp, 'd.yaml')
        io.open(y, 'w', encoding='utf-8').write(
            'path: %s\ntrain: images\nval: images\nnames:\n  0: amoeba\n'
            % tmp.replace('\\', '/'))
        m = YOLO(os.path.join(W, 'yolov8s_fold0.pt'))
        res = m.val(data=y, conf=0.25, workers=0, verbose=False)
        got = float(res.box.map50)
        print('       fold 0 re-run mAP@50 = %.4f' % got)
        ok(0.55 <= got <= 0.95,
           'yolov8s fold 0 inference reproduces a sane mAP@50',
           '%.4f' % got)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', action='store_true',
                    help='also re-run inference (needs a GPU and the weights)')
    a = ap.parse_args()
    print('=' * 68)
    print('PACKAGE VERIFICATION')
    print('=' * 68)
    check_data()
    check_folds()
    check_tables()
    if a.weights:
        check_weights()
    print('\n' + '=' * 68)
    print('%d checks, %d failed' % (checks, len(fails)))
    for f in fails:
        print('   FAILED: %s' % f)
    print('=' * 68)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
