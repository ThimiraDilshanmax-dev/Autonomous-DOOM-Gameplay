"""Full offline + live pipeline check (WSL, CPU torch)."""
import sys
sys.path.insert(0, "/mnt/c/Users/Admin/Desktop/SNN")
import numpy as np, torch
print("torch", torch.__version__)

# 1. snntorch present?
import snntorch as snn
print("snntorch OK", snn.__version__ if hasattr(snn, "__version__") else "imported")

# 2. ANN DQN loss runs
from src.models.ann_mlp import AnnMlp
from src.train.train_ann_basic import ReplayBuffer, dqn_loss_fn
q = AnnMlp(784, 128, 3); t = AnnMlp(784, 128, 3)
buf = ReplayBuffer(500)
rng = np.random.default_rng(0)
for _ in range(200):
    s = rng.random(784, dtype=np.float32); ns = rng.random(784, dtype=np.float32)
    buf.push(s, int(rng.integers(0, 3)), float(rng.random()), ns, 0.0)
loss = dqn_loss_fn(q, t, buf.sample(32))
print("dqn_loss OK", float(loss))

# 3. SNN forward with snnTorch backend + neuron budget
from src.models.snn_mlp import SnnMlp
import src.models.snn_mlp as m
print("snn backend snnTorch:", m._HAS_SNNTORCH)
snn_net = SnnMlp(784, 128, 3, T=8, beta=0.5)
out = snn_net(torch.rand(8, 2, 784))
print("snn forward OK", tuple(out.shape), "neurons=", snn_net.neuron_count())
assert snn_net.neuron_count() <= 1024

# 4. Poisson + export bundle
from src.env.poisson_encoder import poisson_encode
from src.export.cerebra_export import export_snn
sp = poisson_encode(np.full((28, 28), 0.8, dtype=np.float32), T=50, seed=0)
print("poisson rate", round(float(sp.mean()), 3))
cfg = export_snn(snn_net, out_dir="/tmp/snn_export_test", beta=0.5, T=8)
print("export OK total_neurons=", cfg["total_neurons"])

# 5. Live ViZDoom short training (2 episodes, proves train loop works)
from src.train.train_ann_basic import run_vizdoom_dqn
net = run_vizdoom_dqn("/mnt/c/Users/Admin/Desktop/SNN/configs/basic.yaml", episodes=2)
print("vizdoom DQN loop OK, param mean=", round(float(net.net[0].weight.mean()), 5))
print("FULL_VERIFY_PASSED")
