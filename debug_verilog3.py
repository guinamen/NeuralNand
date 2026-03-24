"""
Debug3: compara exatamente o pool usado no forward da rede
com o pool construído manualmente no verilog_gen.
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
x = X[0].unsqueeze(0)  # 0+0=0

print("Comparando pool do forward (rede) vs pool do verilog_gen (manual)")
print("="*60)

# ── Executa forward capturando pools reais ─────────────────────
model.eval()
with torch.no_grad():
    h = model.layers[0](x, gamma)
    real_pools = [x]        # pool da camada 1 = entradas
    real_outs  = [h]

    for layer in model.layers[1:]:
        pool = torch.cat([h, x], dim=1)
        real_pools.append(pool)
        h = layer(pool, gamma)
        real_outs.append(h)

# ── Reconstrói pool manualmente como verilog_gen faz ───────────
base_sigs = [f'x{k}' for k in range(meta.bits_in)]
x_vals    = {f'x{k}': x[0,k].item() for k in range(meta.bits_in)}
prev_sigs = list(base_sigs)
prev_vals = dict(x_vals)

n_layers = len(model.layers)
print(f"\n{'camada':<8} {'neuronio':<10} {'i*':<6} {'j*':<6} "
      f"{'pool[i*] manual':<20} {'pool[i*] real':<20} {'match?'}")
print("-"*72)

for li, layer in enumerate(model.layers):
    is_output = (li == n_layers - 1)

    # Pool manual (como verilog_gen)
    pool_sigs = list(base_sigs) if li == 0 else prev_sigs + base_sigs
    pool_vals_manual = {s: x_vals[s] for s in base_sigs} if li == 0 else \
                       {**{s: prev_vals.get(s, 0) for s in prev_sigs}, **x_vals}

    # Pool real (do forward)
    pool_real = real_pools[li]  # tensor (1, pool_size)

    if len(pool_sigs) != pool_real.shape[1]:
        print(f"  ERRO: pool manual={len(pool_sigs)} != pool real={pool_real.shape[1]}")
        continue

    layer_new_sigs = []
    layer_new_vals = {}

    for ni, neuron in enumerate(layer.neurons):
        i_star, j_star = neuron.crisp_connections(gamma_synth=1000.0)
        i_safe = min(i_star, len(pool_sigs)-1)
        j_safe = min(j_star, len(pool_sigs)-1)

        # Valor manual
        val_a_manual = pool_vals_manual.get(pool_sigs[i_safe], float('nan'))
        val_b_manual = pool_vals_manual.get(pool_sigs[j_safe], float('nan'))

        # Valor real
        val_a_real = pool_real[0, i_safe].item()
        val_b_real = pool_real[0, j_safe].item()

        match_a = abs(val_a_manual - val_a_real) < 0.001
        match_b = abs(val_b_manual - val_b_real) < 0.001
        ok = "OK" if (match_a and match_b) else "DIVERGE"

        if ok == "DIVERGE" or ni < 4:
            tag = 'saida' if is_output else f'L{li+1}'
            print(f"  {tag:<6}  N{ni:<8}  {i_safe:<6} {j_safe:<6}  "
                  f"{pool_sigs[i_safe]}={val_a_manual:.3f} vs {val_a_real:.3f}  "
                  f"{pool_sigs[j_safe]}={val_b_manual:.3f} vs {val_b_real:.3f}  {ok}")

        # Calcula saída para proximo pool
        out_name = f'y{ni}' if is_output else f'l{li+1}_n{ni}'
        bool_out = 1 - val_a_manual * val_b_manual
        layer_new_sigs.append(out_name)
        layer_new_vals[out_name] = bool_out

    prev_sigs = layer_new_sigs
    prev_vals = layer_new_vals

print()
print("Se todos os valores mostram OK, o pool está correto.")
print("Se houver DIVERGE, o pool manual não corresponde ao real.")
