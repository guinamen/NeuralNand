"""
Diagnóstico minimal: verifica se o grafo extraído da rede
produz as mesmas saídas que a rede contínua com gamma_crisp.

Simula o circuito booleano puro em Python — sem envolver Verilog —
e compara com a rede neural.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as F
from nand_net import NANDNet, AnnealingConfig, generate, NANDTrainer, TrainerConfig

torch.manual_seed(42)
X, Y, meta = generate('add', n=2)

annealing = AnnealingConfig(
    gamma_0=0.5, gamma_max=20.0, tau_sched=800.0,
    gamma_crisp=19.0, sharpness=10.0, kappa=4.0,
)
model = NANDNet(n_inputs=meta.bits_in, n1=16, n_layers=4, n_outputs=meta.bits_out)
trainer = NANDTrainer(
    model=model, meta=meta, annealing=annealing,
    config=TrainerConfig(epochs=3000, epsilon=0.05, delta_weights=4.0, log_every=1000),
)
trainer.fit(X, Y)

gamma = annealing.gamma_crisp
print(f"\n=== Diagnostico de Extracao do Grafo ===")
print(f"gamma_crisp = {gamma}")

# ── Avaliação da rede contínua ───────────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred_cont = model(X, gamma)
    pred_bin  = (pred_cont >= 0.5).float()
    acc = (pred_bin == Y).all(dim=1).float().mean().item()
print(f"Rede continua acc_row={acc:.2%}")

# ── Simula o circuito booleano puro em Python ────────────────────────────────
# Reconstrói o grafo exatamente como verilog_gen faria
base_sigs  = list(range(meta.bits_in))  # indices 0..3 = x0..x3

def nand_bool(a, b):
    return 1 - (a * b)

errors = 0
print(f"\n{'entrada':<10} {'esperado':<10} {'continuo':<12} {'bool_sim':<12} {'ok?'}")
print("-" * 52)

for row in range(len(X)):
    x_vals = {f'x{k}': int(X[row, k].item()) for k in range(meta.bits_in)}

    # Simula forward booleano camada a camada
    prev_sigs_vals = dict(x_vals)   # começa com entradas
    base_vals      = dict(x_vals)

    n_layers = len(model.layers)
    for li, layer in enumerate(model.layers):
        is_output = (li == n_layers - 1)
        pool_vals = dict(prev_sigs_vals) if li == 0 else {**prev_sigs_vals, **base_vals}
        pool_list = list(pool_vals.keys())

        new_vals = {}
        for ni, neuron in enumerate(layer.neurons):
            i_star, j_star = neuron.crisp_connections(gamma_synth=1000.0)
            i_safe = min(i_star, len(pool_list) - 1)
            j_safe = min(j_star, len(pool_list) - 1)
            sig_a  = pool_list[i_safe]
            sig_b  = pool_list[j_safe]
            a_val  = pool_vals[sig_a]
            b_val  = pool_vals[sig_b]

            out_val = nand_bool(a_val, b_val)
            out_name = f'y{ni}' if is_output else f'l{li+1}_n{ni}'
            new_vals[out_name] = out_val

        prev_sigs_vals = new_vals

    # Saídas booleanas
    y_bool = [new_vals[f'y{k}'] for k in range(meta.bits_out)]
    y_cont = [int((pred_cont[row, k] >= 0.5).item()) for k in range(meta.bits_out)]
    y_true = [int(Y[row, k].item()) for k in range(meta.bits_out)]

    a = int(X[row,0]) + int(X[row,1])*2
    b = int(X[row,2]) + int(X[row,3])*2
    exp_v  = sum(y_true[k]*(2**k) for k in range(meta.bits_out))
    cont_v = sum(y_cont[k]*(2**k) for k in range(meta.bits_out))
    bool_v = sum(y_bool[k]*(2**k) for k in range(meta.bits_out))

    ok = "OK" if bool_v == exp_v else "FAIL"
    if ok == "FAIL":
        errors += 1
    print(f"  {a}+{b}={exp_v:<7}  cont={cont_v:<5}  bool={bool_v:<5}  {ok}")

print(f"\nSimulacao booleana: {errors} erros / {len(X)}")
print()
print("Concentracao do softmax (gamma_synth=19 vs 1000):")
print(f"{'neuronio':<12} {'conc_a@19':<14} {'conc_b@19':<14} {'conc_a@1000':<14} {'conc_b@1000'}")
print("-" * 62)
for ni, neuron in enumerate(model.layers[0].neurons[:8]):
    ca19, cb19 = neuron.softmax_concentration(1/19)
    ca1k, cb1k = neuron.softmax_concentration(1/1000)
    flag = " <-- PROBLEMA" if min(ca19,cb19) < 0.99 else ""
    print(f"  L1_N{ni:<7} {ca19:<14.4f} {cb19:<14.4f} {ca1k:<14.6f} {cb1k:.6f}{flag}")
print()

# ── Mostra valores intermediários para a primeira entrada com erro ────────────
print("Inspecionando valores intermediarios na fronteira (entrada 0+0=0):")
x_vals = {f'x{k}': int(X[0, k].item()) for k in range(meta.bits_in)}
base_vals = dict(x_vals)
prev_vals = dict(x_vals)

for li, layer in enumerate(model.layers):
    is_output = (li == n_layers - 1)
    pool_vals = dict(prev_vals) if li == 0 else {**prev_vals, **base_vals}
    pool_list = list(pool_vals.keys())
    new_vals = {}

    for ni, neuron in enumerate(layer.neurons):
        i_star, j_star = neuron.crisp_connections(gamma_synth=1000.0)
        i_safe = min(i_star, len(pool_list) - 1)
        j_safe = min(j_star, len(pool_list) - 1)
        sig_a, sig_b = pool_list[i_safe], pool_list[j_safe]
        a_val, b_val = pool_vals[sig_a], pool_vals[sig_b]
        out_val  = nand_bool(a_val, b_val)
        out_name = f'y{ni}' if is_output else f'l{li+1}_n{ni}'
        new_vals[out_name] = out_val

        # Saida continua do mesmo neuronio
        with torch.no_grad():
            pool_t = torch.tensor(list(pool_vals.values()), dtype=torch.float32).unsqueeze(0)
            cont_v = neuron(pool_t, gamma).item()

        if abs(out_val - cont_v) > 0.1:
            print(f"  DIVERGENCIA L{li+1}_N{ni}: "
                  f"NAND({sig_a}={a_val},{sig_b}={b_val}) "
                  f"bool={out_val:.0f}  cont={cont_v:.4f}")

    prev_vals = new_vals

print("\nSaidas finais (0+0=0):", {k: v for k,v in new_vals.items()})
