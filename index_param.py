from typing import Tuple

procs = [16, 128, 1024]
xs = [20, 50, 100, 200]
ts = [20, 50, 100, 200]
rs = [2, 5, 10, 20]
num_trials = 20

# converts the job index to the parameters
# the parameters are x, y, t, r, p, trial
def index_to_params(index: int) -> Tuple[int, int, int, int, int, int]:
    num = index - 1
    num, trial = divmod(num, num_trials)
    num, index = divmod(num, len(procs))
    p = procs[index]
    num, index = divmod(num, len(rs))
    r = rs[index]
    num, index = divmod(num, len(ts))
    t = ts[index]
    num, index = divmod(num, len(xs))
    x = xs[index]
    return x, x, t, r, p, (trial+1)

# converts the parameters to job index
def params_to_index(x: int, t: int, r: int, p: int, trial: int) -> int:
    num = xs.index(x)
    num = num *  len(ts) + ts.index(t)
    num = num * len(rs) + rs.index(r)
    num = num * len(procs) + procs.index(p)
    num = num * num_trials + trial
    return num

x = 50
t = 20
r = 2
p = 16
start = params_to_index(x, t, r, p, 1)
end = params_to_index(x, t, r, p, 20)
print(f'{start} - {end}')