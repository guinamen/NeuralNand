"""
Diagnostico 2: entende por que NAND(l1_n4, l1_n14) da cont=0.9994
quando as entradas sao 0+0=0 (x0=x1=x2=x3=0).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import torch, torch.nn.functional as F
from nand_net import NANDNet, AnnealingConfig, generate, NANDTrainer, TrainerConfig

torch.manual_seed(42)
X, Y, meta = generate('add', n=2)
annealing = AnnealingConfig(gamma_0=0.5, gamma_max=35.0, tau_sched=800.0,
                             gamma_crisp=30.0, sharpness=10.0, kappa=4.0)
model = NANDNet(n_inputs=meta.bits_in, n1=16, n_layers=4, n_outputs=meta.bits_out)
trainer = NANDTrainer(model=model, meta=meta, annealing=annealing,
    config=TrainerConfig(epochs=3000, epsilon=0.05, delta_weights=4.0, log_every=5000))
trainer.fit(X, Y)

gamma = annealing.gamma_crisp
x = X[0]  # 0+0=0: x0=x1=x2=x3=0
print(f"Entrada: {x.tolist()} (0+0=0)")
print()

# Captura saidas reais da camada 1
model.eval()
with torch.no_grad():
    layer1_out = model.layers[0](x.unsqueeze(0), gamma).squeeze()

print("Saidas reais da camada 1 (continuas):")
for i, v in enumerate(layer1_out):
    print(f"  l1_n{i} = {v.item():.6f}")

print()

# Agora mostra o que a simulacao booleana assume
print("O que a simulacao booleana ASSUME (argmax one-hot):")
base_sigs = [f'x{k}' for k in range(4)]
x_vals = {f'x{k}': int(x[k].item()) for k in range(4)}
pool = list(x_vals.keys())

for ni, neuron in enumerate(model.layers[0].neurons):
    i_star, j_star = neuron.crisp_connections(gamma_synth=1000.0)
    i_safe = min(i_star, len(pool)-1)
    j_safe = min(j_star, len(pool)-1)
    a_val = x_vals[pool[i_safe]]
    b_val = x_vals[pool[j_safe]]
    bool_out = 1 - a_val * b_val  # NAND booleano
    cont_out = layer1_out[ni].item()
    diff = abs(bool_out - cont_out)
    flag = " <-- DIVERGE" if diff > 0.05 else ""
    print(f"  l1_n{ni}: NAND({pool[i_safe]}={a_val}, {pool[j_safe]}={b_val}) "
          f"= {bool_out}  cont={cont_out:.4f}  diff={diff:.4f}{flag}")

print()
print("CONCLUSAO:")
print("Se os valores booleanos de l1_ni divergem dos continuos,")
print("todas as camadas seguintes tambem divergem — efeito cascata.")
print()
print("Solucao: a sintese Verilog deve usar os valores que a rede")
print("realmente computa, nao NAND(0,1) ideal. Isso significa que")
print("o circuito nao pode ser expresso como NAND primitivo a menos")
print("que os sinais intermediarios sejam exatamente 0 ou 1.")
print()

# Verifica se os sinais intermediarios sao suficientemente binarios
print("Verificacao de binaridade dos sinais intermediarios (entrada 0+0=0):")
with torch.no_grad():
    h = layer1_out.unsqueeze(0)
    layers_out = [h]
    for layer in model.layers[1:]:
        pool_t = torch.cat([h, x.unsqueeze(0)], dim=1)
        h = layer(pool_t, gamma)
        layers_out.append(h)

for li, out in enumerate(layers_out):
    vals = out.squeeze()
    frontier = [(i, v.item()) for i, v in enumerate(vals) if 0.05 < v.item() < 0.95]
    tag = 'saida' if li == len(layers_out)-1 else f'L{li+1}'
    if frontier:
        print(f"  {tag}: {len(frontier)} valores na fronteira:")
        for idx, v in frontier:
            print(f"    [{idx}] = {v:.4f}")
    else:
        print(f"  {tag}: BINARIO (todos < 0.05 ou > 0.95)")
