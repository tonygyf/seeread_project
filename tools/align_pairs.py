"""QC for tripod-captured (dark, torch-lit) pairs: ECC alignment residual. Pairs with mean |shift| > tol px are flagged,
so polygons annotated on the lit frame can be reused on the dark frame.
python tools/align_pairs.py --dark d.jpg --lit l.jpg"""
import argparse, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dark"); ap.add_argument("--lit"); ap.add_argument("--tol", type=float, default=2.0)
a = ap.parse_args()
d, l = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY).astype(np.float32) for p in (a.dark, a.lit)]
d = cv2.normalize(cv2.GaussianBlur(d, (0, 0), 2), None, 0, 1, cv2.NORM_MINMAX); l = cv2.normalize(cv2.GaussianBlur(l, (0, 0), 2), None, 0, 1, cv2.NORM_MINMAX)
warp = np.eye(2, 3, dtype=np.float32)
cc, warp = cv2.findTransformECC(l, d, warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
shift = float(np.hypot(*warp[:, 2])); print(f"ECC={cc:.3f} shift={shift:.2f}px", "OK" if shift <= a.tol else "FLAG")
