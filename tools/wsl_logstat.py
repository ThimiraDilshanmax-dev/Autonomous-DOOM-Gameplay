import re, sys
log = sys.argv[1] if len(sys.argv) > 1 else "/home/thimira/basic1000.log"
lines = open(log).read().splitlines()
p = re.compile(r"ep=(\d+) reward=([-\d.]+).*steps=(\d+)")
eps, pos = 0, 0
recent, steps = [], []
for ln in lines:
    m = p.search(ln)
    if m:
        eps += 1
        r = float(m.group(2))
        s = int(m.group(3)) if m.group(3) else 0
        recent.append(r)
        steps.append(s)
        if r > 0:
            pos += 1
recent = recent[-50:]
steps = steps[-50:]
print(f"episodes={eps} kills={pos} kill_rate={pos/max(eps,1):.3f}")
if recent:
    print(f"last50 mean={sum(recent)/len(recent):.1f} max={max(recent):.1f}")
if steps:
    print(f"last50 survival mean_steps={sum(steps)/len(steps):.1f} "
          f"max_steps={max(steps)} (use for health_gathering)")
