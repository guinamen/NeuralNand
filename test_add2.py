"""
test_add2.py — Somador de 2 bits via NAND Neural Network.

Estrategia correta para sintese Verilog:
  1. Treinar com gamma_max alto e lambda_max forte
  2. lambda forte garante w1,w2 -> 1.0
  3. Com w1=w2=1.0, z = a+b-1.5 esta longe de zero para entradas binarias
  4. Qualquer gamma >= 15 entao binariza perfeitamente os nos intermediarios
"""

import sys, os, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from nand_net import (
    NANDNet, AnnealingConfig,
    NANDTrainer, TrainerConfig,
    generate, describe, VerilogGenerator,
)

torch.manual_seed(42)

# ── Dataset ──────────────────────────────────────────────────────────────────
X, Y, meta = generate('add', n=2)
print('=' * 60)
print('NAND Neural Network — Somador 2 bits')
print('=' * 60)
print(describe(meta))
print()

print('Tabela verdade (primeiras 8 linhas):')
print(f'  {"a1 a0 b1 b0":<14}  {"s2 s1 s0":<10}  expressao')
print('  ' + '-' * 40)
for i in range(min(8, len(X))):
    ins  = ' '.join(str(int(b)) for b in X[i])
    outs = ' '.join(str(int(b)) for b in Y[i])
    a = int(X[i,0]) + int(X[i,1])*2
    b = int(X[i,2]) + int(X[i,3])*2
    s = int(Y[i,0]) + int(Y[i,1])*2 + int(Y[i,2])*4
    print(f'  {ins:<14}  {outs:<10}  {a} + {b} = {s}')
print()

# ── Modelo ────────────────────────────────────────────────────────────────────
model = NANDNet(n_inputs=meta.bits_in, n1=16, n_layers=4, n_outputs=meta.bits_out)
print(model.summary())
print()

# ── Annealing ─────────────────────────────────────────────────────────────────
# lambda_max=0.1: regularizacao forte garante w1,w2 -> 1.0
# Com w=1.0, z = 1+1-1.5 = 0.5 (longe de zero), qualquer gamma binariza
annealing = AnnealingConfig(
    gamma_0     = 0.5,
    gamma_max   = 25.0,
    tau_sched   = 800.0,
    gamma_crisp = 20.0,
    sharpness   = 10.0,
    kappa       = 2.0,
    lambda_max  = 0.1,    # chave: regularizacao forte -> w -> 1.0
)

# ── Verificacao de inicializacao ──────────────────────────────────────────────
with torch.no_grad():
    pred_0 = model(X[:4], annealing.gamma_0)
    z_check = pred_0.mean().item()
print(f'Verificacao init: saida media = {z_check:.4f}  (esperado ~0.5)')
print()

# ── Treino ────────────────────────────────────────────────────────────────────
trainer = NANDTrainer(
    model     = model,
    meta      = meta,
    annealing = annealing,
    config    = TrainerConfig(
        epochs        = 4000,
        epsilon       = 0.05,
        delta_weights = 4.0,
        log_every     = 200,
    ),
)

print('Iniciando treino...')
print(f'{"epoca":<6} {"gamma":<10} {"fase":<10} '
      f'{"alpha":<6} {"beta":<6} {"arith":<10} {"acc_row":<8}')
print('-' * 56)
history = trainer.fit(X, Y)

# ── Verificacao de pesos ──────────────────────────────────────────────────────
print()
print('Pesos w1, w2 apos treino (devem estar proximos de 1.0):')
all_w = []
for li, layer in enumerate(model.layers):
    for ni, neuron in enumerate(layer.neurons):
        w1, w2 = neuron.crisp_weights()
        all_w.extend([w1, w2])
        if li == 0 and ni < 4:
            print(f'  L{li+1}_N{ni}: w1={w1:.4f}  w2={w2:.4f}  '
                  f'z(1,1)={w1+w2-1.5:+.4f}  {"OK" if abs(w1-1)<0.15 and abs(w2-1)<0.15 else "LONGE de 1"}')
w_mean = sum(all_w)/len(all_w)
w_max_dev = max(abs(w-1.0) for w in all_w)
print(f'  Media global: {w_mean:.4f}  Desvio max de 1.0: {w_max_dev:.4f}')

# ── Avaliacao ─────────────────────────────────────────────────────────────────
def print_result(result, label):
    print(f'\n  [{label}]')
    print(f'  Acuracia : {result["acc_row"]:.2%}  ({result["n_correct"]}/{result["n_total"]})')
    print(f'  L_arith  : {result["loss_arith"]:.6f}')
    print(f'  Perfeito : {"SIM" if result["is_perfect"] else "NAO"}')
    if result['errors']:
        print(f'  Erros ({len(result["errors"])}):', end='')
        for e in result['errors'][:5]:
            a = int(e['input'][0]) + int(e['input'][1])*2
            b = int(e['input'][2]) + int(e['input'][3])*2
            got = sum(int(e['got'][k])*(2**k) for k in range(len(e['got'])))
            exp = sum(int(e['expected'][k])*(2**k) for k in range(len(e['expected'])))
            print(f'  {a}+{b}={exp}→{got}', end='')
        print()

print()
print('=' * 60)
print('Avaliacao:')
r = trainer.evaluate(X, Y)
print_result(r, f'gamma_crisp={annealing.gamma_crisp}')

# ── Verificacao de binaridade dos sinais intermediarios ───────────────────────
print()
print('Binaridade dos sinais intermediarios com gamma_crisp:')
gen_check = VerilogGenerator(model=model, meta=meta,
    annealing=annealing, module_name=f'nand_{meta.op}{meta.n}')
v = gen_check.verify_binary(X, gamma=annealing.gamma_crisp)
if v['is_binary_clean']:
    print('  LIMPO — todos os nos sao binarios, sintese segura')
else:
    print(f'  {v["total_frontier"]} nos na fronteira por camada: {v["by_layer"]}')
    print('  Aumentar lambda_max ou gamma_max e re-treinar')

# ── Geracao Verilog ───────────────────────────────────────────────────────────
print()
print('=' * 60)
print('Gerando Verilog...')

gen = VerilogGenerator(model=model, meta=meta,
    annealing=annealing, module_name=f'nand_{meta.op}{meta.n}')
s = gen.stats()
print(f'Circuito: {s["total_raw"]} raw -> {s["total_gates"]} apos poda')
print(f'  NAND: {s["nand_gates"]}  NOT: {s["not_gates"]}  removidas: {s["pruned_dead"]}')

out_dir = os.path.join(os.path.dirname(__file__), 'verilog_out')
paths = gen.write(out_dir, X, Y)
print()
for kind, path in paths.items():
    print(f'  {kind:<12} {path}')

print()
print('=' * 60)
print('Modulo Verilog:')
print('=' * 60)
print(gen.generate_module())
