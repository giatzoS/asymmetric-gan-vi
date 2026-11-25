import sys, glob, json, re, numpy as np
runs = {}
for root in sys.argv[1:]:
    for p in glob.glob(root + "/**/metrics.json", recursive=True):
        m = re.search(r"/([^/]+)/([^/]+)_seed\d+/metrics\.json$", p)
        if m:
            runs.setdefault((m.group(1), m.group(2)), []).append(json.load(open(p)))
keys = [("frechet_inception_distance", "FID"),
        ("inception_score_mean", "IS"),
        ("kernel_inception_distance_mean", "KID")]
for (exp, meth), ms in sorted(runs.items()):
    print(exp, meth, "n =", len(ms))
    for k, n in keys:
        v = np.array([x[k] for x in ms if k in x])
        if len(v):
            print("   %-4s %.4f  std(ddof1) %.4f  std(ddof0) %.4f" % (n, v.mean(), v.std(ddof=1), v.std(ddof=0)))
