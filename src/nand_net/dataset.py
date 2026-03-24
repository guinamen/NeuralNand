"""
dataset.py
----------
Gerador de tabela verdade a partir de operadores aritmeticos.

Convencoes:
  - Bit ordering: LSB-first em ambos os lados
  - Pesos da loss: w_k = 2^k / (2^bits_out - 1)  (calibrado por posicao)
  - Sub usa complemento de 2 com n+1 bits (bit MSB = sinal)
  - Mul preserva produto completo: 2n bits
  - Todos os bits de saida sao preservados (sem truncamento)
"""

import torch
from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict


@dataclass
class DatasetMeta:
    op:        str
    n:         int          # bits por operando
    bits_in:   int
    bits_out:  int
    encoding:  str
    n_rows:    int
    weights:   torch.Tensor  # (bits_out,) pesos da loss ponderada


def _to_bits_unsigned(val: int, n: int) -> list:
    return [(val >> k) & 1 for k in range(n)]

def _to_bits_signed(val: int, n: int) -> list:
    """Complemento de 2 com n bits."""
    mask = (1 << n) - 1
    v    = val & mask
    return [(v >> k) & 1 for k in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Definicao dos operadores
# ─────────────────────────────────────────────────────────────────────────────

def _make_ops() -> Dict:
    """
    Cada operador define:
        arity    : 1 (unario) ou 2 (binario)
        out_bits : funcao n -> bits_out
        encoding : descricao do encoding de saida
        fn       : (a, b, n) -> lista de bits de saida (LSB-first)
    """
    return {
        'add': dict(
            arity=2, encoding='unsigned',
            out_bits=lambda n: n + 1,
            fn=lambda a, b, n: _to_bits_unsigned(a + b, n + 1),
        ),
        'sub': dict(
            arity=2, encoding='complemento_2',
            out_bits=lambda n: n + 1,
            fn=lambda a, b, n: _to_bits_signed(a - b, n + 1),
        ),
        'mul': dict(
            arity=2, encoding='unsigned',
            out_bits=lambda n: 2 * n,
            fn=lambda a, b, n: _to_bits_unsigned(a * b, 2 * n),
        ),
        'inc': dict(
            arity=1, encoding='unsigned',
            out_bits=lambda n: n + 1,
            fn=lambda a, _, n: _to_bits_unsigned(a + 1, n + 1),
        ),
        'abs': dict(
            arity=1, encoding='unsigned',
            out_bits=lambda n: n,
            fn=lambda a, _, n: _to_bits_unsigned(a, n),
        ),
        'cmp': dict(
            arity=2, encoding='bool',
            out_bits=lambda n: 1,
            fn=lambda a, b, _: [1 if a == b else 0],
        ),
        'max': dict(
            arity=2, encoding='unsigned',
            out_bits=lambda n: n,
            fn=lambda a, b, n: _to_bits_unsigned(max(a, b), n),
        ),
        'min': dict(
            arity=2, encoding='unsigned',
            out_bits=lambda n: n,
            fn=lambda a, b, n: _to_bits_unsigned(min(a, b), n),
        ),
    }

OPERATORS = _make_ops()


# ─────────────────────────────────────────────────────────────────────────────
# Gerador principal
# ─────────────────────────────────────────────────────────────────────────────

def generate(op: str, n: int) -> Tuple[torch.Tensor, torch.Tensor, DatasetMeta]:
    """
    Gera a tabela verdade completa para o operador op com n bits por operando.

    Args:
        op : nome do operador ('add', 'sub', 'mul', 'inc', 'abs', 'cmp', 'max', 'min')
        n  : bits por operando (recomendado: 2 a 5)

    Returns:
        X    : (n_rows, bits_in)   float32 tensor em {0.0, 1.0}
        Y    : (n_rows, bits_out)  float32 tensor em {0.0, 1.0}
        meta : DatasetMeta com informacoes do dataset e pesos da loss
    """
    if op not in OPERATORS:
        raise ValueError(f"Operador '{op}' desconhecido. Disponiveis: {list(OPERATORS)}")

    spec     = OPERATORS[op]
    arity    = spec['arity']
    bits_out = spec['out_bits'](n)
    bits_in  = 2 * n if arity == 2 else n
    max_a    = 1 << n
    max_b    = (1 << n) if arity == 2 else 1

    rows_x, rows_y = [], []
    for a in range(max_a):
        for b in range(max_b):
            in_bits  = _to_bits_unsigned(a, n)
            if arity == 2:
                in_bits += _to_bits_unsigned(b, n)
            out_bits = spec['fn'](a, b, n)
            rows_x.append(in_bits)
            rows_y.append(out_bits)

    X = torch.tensor(rows_x, dtype=torch.float32)
    Y = torch.tensor(rows_y, dtype=torch.float32)

    # Pesos da loss ponderada: w_k = 2^k / (2^bits_out - 1)
    denom   = (1 << bits_out) - 1 if bits_out > 0 else 1
    weights = torch.tensor(
        [(1 << k) / denom for k in range(bits_out)],
        dtype=torch.float32
    )

    meta = DatasetMeta(
        op       = op,
        n        = n,
        bits_in  = bits_in,
        bits_out = bits_out,
        encoding = spec['encoding'],
        n_rows   = len(rows_x),
        weights  = weights,
    )
    return X, Y, meta


def describe(meta: DatasetMeta) -> str:
    """Imprime um resumo legivel do dataset."""
    lines = [
        f'Operador : {meta.op}   n_bits : {meta.n}   encoding : {meta.encoding}',
        f'Entradas : {meta.bits_in} bits   Saidas : {meta.bits_out} bits   '
        f'Linhas : {meta.n_rows}',
        f'Pesos loss (LSB->MSB): {meta.weights.tolist()}',
    ]
    return '\n'.join(lines)
