from tqdm import tqdm, trange
import time
from zeros.act_func import _liner

for _ in trange(100):
    time.sleep(0.01)

for _ in tqdm(range(100)):
    time.sleep(0.01)


class T:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        pass
