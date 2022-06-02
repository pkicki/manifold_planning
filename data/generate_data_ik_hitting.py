import sys
import os
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.dirname(parentdir))

from data.hpo import get_hitting_configuration_opt

urdf_path = os.path.join(os.path.dirname(__file__), "../iiwa_striker_new.urdf")

data = []
ds = sys.argv[1]
assert ds in ["train", "val", "test"]
idx = int(sys.argv[2])
N = int(sys.argv[3])

xl = 0.55
xh = 1.45
yl = -0.45
yh = 0.45
for i in range(N):
    x = (xh - xl) * np.random.rand() + xl
    y = (yh - yl) * np.random.rand() + yl
    th = np.pi * (2 * np.random.random() - 1.)
    q, *_ = get_hitting_configuration_opt(x, y, th)
    if q is None:
        i -= 1
        continue
    data.append([x, y, th] + q)

dir_name = f"paper/ik_hitting/{ds}"
os.makedirs(dir_name, exist_ok=True)
np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
os.popen(f'cp {os.path.basename(__file__)} {dir_name}')