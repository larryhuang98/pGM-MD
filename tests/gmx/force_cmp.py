import os, sys, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dd_compare.py")).read().split("args = sys.argv[1:]")[0])
_, _, Fd = read_trr_first(sys.argv[1]); _, _, Fg = read_trr_first(sys.argv[2])
d = Fg - Fd
print("GPU (single) vs CPU (double) forces: max |dF| %.3e, rms dF / rms F %.2e, max |F| %.1f" % (np.abs(d).max(), np.sqrt((d**2).mean()/(Fd**2).mean()), np.abs(Fd).max()))
