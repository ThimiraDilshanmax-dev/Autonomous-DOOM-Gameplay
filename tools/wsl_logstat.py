import re, sys
log = sys.argv[1] if len(sys.argv) > 1 else "/home/thimira/basic1000.log"
lines = open(log).read().splitlines()
p = re.compile(r"ep=(\d+) reward=([-\d.]+)")
eps, pos = 0, 0
recent = []
for ln in lines:
    m = p.search(ln)
    if m:
        eps += 1
        r = float(m.group(2))
        recent.append(r)
        if r > 0:
            pos += 1
recent = recent[-50:]
print(f"episodes={eps} kills={pos} kill_rate={pos/max(eps,1):.3f}")
print(f"last50 mean={sum(recent)/len(recent):.1f} max={max(recent):.1f}")
