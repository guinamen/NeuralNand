"""
verilog_gen.py
--------------
Gera circuito Verilog sintetizavel a partir de uma NANDNet cristalizada.

Producoes:
  <nome>.v      -- modulo Verilog otimizado (sem portas mortas, sem duplicatas)
  <nome>_tb.v   -- testbench que verifica toda a tabela verdade

Otimizacoes aplicadas antes da emissao:
  1. Deduplicacao : portas com mesmas entradas {a,b} sao fundidas em uma so
  2. Alcancabilidade: portas cujos sinais nao chegam a nenhuma saida sao removidas

Convencoes:
  - Sinais de entrada : x0..x{n_in-1}  (LSB-first)
  - Sinais de saida   : y0..y{n_out-1} (LSB-first)
  - Wires internos    : l{camada}_n{neuronio}
  - gamma de sintese  : gamma_crisp (nao gamma_max)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

from .nand_net import NANDNet, AnnealingConfig
from .dataset  import DatasetMeta


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Gate:
    name:   str
    out:    str
    in_a:   str
    in_b:   str
    is_not: bool
    layer:  int
    neuron: int
    w1:     float
    w2:     float


# ─────────────────────────────────────────────────────────────────────────────
# Gerador
# ─────────────────────────────────────────────────────────────────────────────

class VerilogGenerator:

    def __init__(
        self,
        model:          NANDNet,
        meta:           DatasetMeta,
        annealing:      AnnealingConfig,
        module_name:    str   = 'nand_circuit',
        gamma_synthesis: float = 200.0,
    ):
        """
        gamma_synthesis: gamma usado para calcular crisp_connections().
            Deve ser muito maior que gamma_crisp para garantir que o
            softmax seja verdadeiramente one-hot.
            Com gamma=200, tau=0.005 -- qualquer diferenca de logits >0.02
            ja produz concentracao >99.9%.
            Default 200 e seguro para todos os casos praticos.
        """
        self.model            = model
        self.meta             = meta
        self.annealing        = annealing
        self.module_name      = module_name
        self.n_in             = meta.bits_in
        self.n_out            = meta.bits_out
        self.gamma_synthesis  = gamma_synthesis

        # Verifica concentracao antes de construir
        self._concentration_report = self._check_concentration()

        raw         = self._build_graph()
        deduped     = self._deduplicate(raw)
        self._gates = self._prune_dead(deduped)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Construcao do grafo bruto
    # ─────────────────────────────────────────────────────────────────────────

    def _check_concentration(self) -> dict:
        """
        Verifica se todos os neuronios tem softmax suficientemente one-hot
        com gamma_synthesis. Retorna relatorio com neuronios problemáticos.
        """
        tau = 1.0 / self.gamma_synthesis
        warnings = []
        min_conc  = 1.0
        for li, layer in enumerate(self.model.layers):
            for ni, neuron in enumerate(layer.neurons):
                ca, cb = neuron.softmax_concentration(tau)
                conc   = min(ca, cb)
                min_conc = min(min_conc, conc)
                if conc < 0.99:
                    warnings.append({
                        'layer': li+1, 'neuron': ni,
                        'conc_a': ca, 'conc_b': cb,
                    })
        return {'min_concentration': min_conc, 'warnings': warnings}

    def _build_graph(self) -> List[Gate]:
        base_sigs = [f'x{k}' for k in range(self.n_in)]
        gates     = []
        prev_sigs = list(base_sigs)
        n_layers  = len(self.model.layers)
        tau_synth = 1.0 / self.gamma_synthesis

        for li, layer in enumerate(self.model.layers):
            is_output = (li == n_layers - 1)
            pool_sigs = list(base_sigs) if li == 0 else prev_sigs + base_sigs
            layer_sigs = []

            for ni, neuron in enumerate(layer.neurons):
                # Usa theta com tau_synthesis para one-hot garantido
                import torch
                import torch.nn.functional as F_
                alpha  = F_.softmax(neuron.theta_a / tau_synth, dim=0)
                beta   = F_.softmax(neuron.theta_b / tau_synth, dim=0)
                i_star = alpha.argmax().item()
                j_star = beta.argmax().item()
                w1, w2         = neuron.crisp_weights()
                i_safe = min(i_star, len(pool_sigs) - 1)
                j_safe = min(j_star, len(pool_sigs) - 1)
                sig_a  = pool_sigs[i_safe]
                sig_b  = pool_sigs[j_safe]
                out    = f'y{ni}' if is_output else f'l{li+1}_n{ni}'

                layer_sigs.append(out)
                gates.append(Gate(
                    name   = f'g_{li+1}_{ni}',
                    out    = out,
                    in_a   = sig_a,
                    in_b   = sig_b,
                    is_not = (i_safe == j_safe),
                    layer  = li + 1,
                    neuron = ni,
                    w1     = w1,
                    w2     = w2,
                ))

            prev_sigs = layer_sigs

        return gates

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Deduplicacao: funde portas com mesmas entradas
    # ─────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, gates: List[Gate]) -> List[Gate]:
        """
        Se duas portas computam NAND(a,b) com {a,b} identicos, a segunda
        e substituida por um alias — seu sinal de saida e remapeado para o
        sinal da primeira porta encontrada com essa funcao.

        NAND e comutativa: NAND(a,b) = NAND(b,a), entao a chave e um
        frozenset das duas entradas.
        """
        # canonical[(frozenset(a,b))] = sinal canônico para essa funcao
        canonical: Dict[frozenset, str] = {}
        # alias[sinal_redundante] = sinal_canonico
        alias: Dict[str, str] = {}

        def resolve(s: str) -> str:
            while s in alias:
                s = alias[s]
            return s

        result = []
        for gate in gates:
            a = resolve(gate.in_a)
            b = resolve(gate.in_b)
            key = frozenset([a, b])

            if key in canonical:
                # Esta porta e redundante — mapeia sua saida para o canonico
                alias[gate.out] = canonical[key]
            else:
                canonical[key] = gate.out
                # Atualiza as entradas com os canonicos resolvidos
                gate.in_a = a
                gate.in_b = b
                gate.is_not = (a == b)
                result.append(gate)

        # Propaga aliases nas entradas das portas restantes
        for gate in result:
            gate.in_a = resolve(gate.in_a)
            gate.in_b = resolve(gate.in_b)
            gate.is_not = (gate.in_a == gate.in_b)

        # Portas de saida podem ter sido aliasadas — precisam ser recriadas
        # se seu sinal canonico aponta para um wire interno
        output_sigs = {f'y{k}' for k in range(self.n_out)}
        final = []
        handled_outputs = set()

        for gate in result:
            # Verifica se alguma saida foi aliasada para este gate
            for yk in sorted(output_sigs - handled_outputs):
                canonical_of_yk = resolve(yk)
                if canonical_of_yk == gate.out and canonical_of_yk != yk:
                    # Cria uma porta NOT(x,x) trivial para renomear o sinal
                    # na verdade usamos assign — mas como so temos NAND,
                    # usamos nand(yk, src, src) que equivale a NOT(src)
                    # e depois NOT(NOT(src)) = src seria necessario.
                    # Solucao mais limpa: alias direto via assign.
                    # Emitimos como comentario no modulo.
                    alias[yk] = canonical_of_yk
                    handled_outputs.add(yk)
            final.append(gate)

        self._output_alias = alias  # guarda para uso no generate_module
        return final

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Poda de portas mortas (analise de alcancabilidade)
    # ─────────────────────────────────────────────────────────────────────────

    def _prune_dead(self, gates: List[Gate]) -> List[Gate]:
        """
        Remove portas cujos sinais de saida nao contribuem para nenhuma saida.

        Algoritmo: BFS reverso a partir dos sinais de saida y0..yn.
        Um sinal e 'vivo' se e alcancado por alguma saida percorrendo
        as arestas do grafo ao contrario.
        """
        alias = getattr(self, '_output_alias', {})

        def resolve(s: str) -> str:
            while s in alias:
                s = alias[s]
            return s

        # Indice: sinal de saida -> gate que o produz
        producer: Dict[str, Gate] = {g.out: g for g in gates}

        # BFS reverso
        live: Set[str] = set()
        queue = [resolve(f'y{k}') for k in range(self.n_out)]

        while queue:
            sig = queue.pop()
            if sig in live or sig.startswith('x'):
                continue
            live.add(sig)
            if sig in producer:
                g = producer[sig]
                queue.append(resolve(g.in_a))
                queue.append(resolve(g.in_b))

        live_gates   = [g for g in gates if g.out in live]
        pruned_count = len(gates) - len(live_gates)

        self._pruned_count = pruned_count
        self._alias        = alias
        return live_gates

    # ─────────────────────────────────────────────────────────────────────────
    # Verificacao de binaridade
    # ─────────────────────────────────────────────────────────────────────────

    def verify_binary(self, X, gamma: float = None, threshold: float = 0.05) -> dict:
        """
        Verifica se todas as saidas intermediarias sao suficientemente binarias
        para que a conversao para Verilog seja correta.

        Um neuronio esta na 'fronteira' se sua saida, com entradas binarias,
        cai em [threshold, 1-threshold] — nem claramente 0 nem claramente 1.
        Neuronios na fronteira produzirao erros no Verilog binario.

        Args:
            X         : tabela de entradas binarias
            gamma     : gamma de avaliacao (default: gamma_crisp)
            threshold : margem de binaridade (default: 0.05)

        Returns:
            dict com contagem de neuronios na fronteira por camada
        """
        import torch
        if gamma is None:
            gamma = self.annealing.gamma_crisp

        self.model.eval()
        frontier_by_layer = {}
        total_frontier = 0

        with torch.no_grad():
            h = self.model.layers[0](X.float(), gamma)
            layers_out = [h]
            for layer in self.model.layers[1:]:
                pool = torch.cat([h, X.float()], dim=1)
                h = layer(pool, gamma)
                layers_out.append(h)

        for li, out in enumerate(layers_out):
            vals = out.flatten()
            frontier = ((vals >= threshold) & (vals <= 1 - threshold)).sum().item()
            frontier_by_layer[f'L{li+1}'] = frontier
            total_frontier += frontier

        is_clean = total_frontier == 0
        return {
            'is_binary_clean': is_clean,
            'total_frontier':  total_frontier,
            'by_layer':        frontier_by_layer,
            'gamma_used':      gamma,
            'threshold':       threshold,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Geracao do modulo Verilog
    # ─────────────────────────────────────────────────────────────────────────

    def generate_module(self) -> str:
        alias  = getattr(self, '_alias', {})

        def resolve(s: str) -> str:
            while s in alias:
                s = alias[s]
            return s

        n_in, n_out = self.n_in, self.n_out
        name        = self.module_name

        pruned = getattr(self, '_pruned_count', 0)
        total_raw = len(self._gates) + pruned

        lines = [
            f'// Gerado por NANDNet',
            f'// Operador    : {self.meta.op}  n_bits : {self.meta.n}',
            f'// Portas raw  : {total_raw}',
            f'// Apos poda   : {len(self._gates)} ({pruned} mortas removidas)',
            f'// gamma_crisp : {self.annealing.gamma_crisp}',
            f'',
        ]

        in_ports  = ', '.join(f'x{k}' for k in range(n_in))
        out_ports = ', '.join(f'y{k}' for k in range(n_out))
        lines += [
            f'module {name} (',
            f'    input  wire {in_ports},',
            f'    output wire {out_ports}',
            f');',
            '',
        ]

        # Wires internos — apenas os vivos e nao-saida
        output_sigs = {f'y{k}' for k in range(n_out)}
        internal = [g.out for g in self._gates if g.out not in output_sigs]
        if internal:
            chunk = 6
            for i in range(0, len(internal), chunk):
                grp = ', '.join(internal[i:i+chunk])
                lines.append(f'    wire {grp};')
            lines.append('')

        # Assigns para saidas que foram aliasadas (deduplicacao)
        for k in range(n_out):
            yk  = f'y{k}'
            src = resolve(yk)
            if src != yk:
                lines.append(f'    assign {yk} = {src};  // deduplicado')
        if any(resolve(f'y{k}') != f'y{k}' for k in range(n_out)):
            lines.append('')

        # Portas por camada
        current_layer = -1
        for gate in self._gates:
            if gate.layer != current_layer:
                current_layer = gate.layer
                n_layers = len(self.model.layers)
                tag = 'saida' if gate.layer == n_layers else f'camada {gate.layer}'
                lines.append(f'    // {tag}')

            a = resolve(gate.in_a)
            b = resolve(gate.in_b)
            comment = f'  // NOT({a})' if gate.is_not else ''
            lines.append(f'    nand {gate.name} ({gate.out}, {a}, {b});{comment}')

        lines += ['', 'endmodule']
        return '\n'.join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Testbench
    # ─────────────────────────────────────────────────────────────────────────

    def generate_testbench(self, X, Y) -> str:
        n_in, n_out = self.n_in, self.n_out
        name        = self.module_name
        lines       = [
            f'// Testbench automatico — {name}',
            f'// {len(X)} vetores de teste',
            f'',
            f'`timescale 1ns/1ps',
            f'',
            f'module {name}_tb;',
            f'',
            f'    reg  ' + ', '.join(f'x{k}' for k in range(n_in)) + ';',
            f'    wire ' + ', '.join(f'y{k}' for k in range(n_out)) + ';',
            f'',
        ]

        conn = ', '.join(
            [f'.x{k}(x{k})' for k in range(n_in)] +
            [f'.y{k}(y{k})' for k in range(n_out)]
        )
        lines += [
            f'    {name} dut ({conn});',
            '',
            '    integer errors = 0;',
            '',
            '    initial begin',
            f'        $display("Testbench: {name}");',
            f'        $display("----------------------------------------");',
            '',
        ]

        for i in range(len(X)):
            x_bits = [int(X[i, k].item()) for k in range(n_in)]
            y_bits = [int(Y[i, k].item()) for k in range(n_out)]

            is_unary = self.meta.op in ('inc', 'abs')
            a_val = sum(x_bits[k] * (2**k) for k in range(n_in if is_unary else n_in // 2))
            b_val = 0 if is_unary else sum(x_bits[n_in//2+k] * (2**k) for k in range(n_in//2))
            y_val = sum(y_bits[k] * (2**k) for k in range(n_out))

            assignments = ' '.join(f'x{k} = {x_bits[k]};' for k in range(n_in))
            lines.append(f'        {assignments} #10;')

            expected = ''.join(str(y_bits[n_out-1-k]) for k in range(n_out))
            got      = '{' + ', '.join(f'y{n_out-1-k}' for k in range(n_out)) + '}'

            if is_unary:
                label_ok   = f'"PASS: {self.meta.op}(%0d)=%0d", {a_val}, {y_val}'
                label_fail = f'"FAIL: {self.meta.op}(%0d)=%0d got=%0d", {a_val}, {y_val}, {got}'
            else:
                label_ok   = f'"PASS: {self.meta.op}(%0d,%0d)=%0d", {a_val}, {b_val}, {y_val}'
                label_fail = f'"FAIL: {self.meta.op}(%0d,%0d)=%0d got=%0d", {a_val}, {b_val}, {y_val}, {got}'

            lines += [
                f'        if ({got} !== {n_out}\'b{expected}) begin',
                f'            $display({label_fail});',
                f'            errors = errors + 1;',
                f'        end else begin',
                f'            $display({label_ok});',
                f'        end',
                '',
            ]

        lines += [
            f'        $display("----------------------------------------");',
            f'        $display("Erros: %0d / {len(X)}", errors);',
            f'        if (errors == 0) $display("CIRCUITO CORRETO");',
            f'        else             $display("CIRCUITO COM FALHAS");',
            f'        $finish;',
            f'    end',
            f'',
            f'endmodule',
        ]
        return '\n'.join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Escrita dos arquivos
    # ─────────────────────────────────────────────────────────────────────────

    def write(self, output_dir: str, X=None, Y=None) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        module_path = os.path.join(output_dir, f'{self.module_name}.v')
        with open(module_path, 'w') as f:
            f.write(self.generate_module())
        paths['module'] = module_path

        if X is not None and Y is not None:
            tb_path = os.path.join(output_dir, f'{self.module_name}_tb.v')
            with open(tb_path, 'w') as f:
                f.write(self.generate_testbench(X, Y))
            paths['testbench'] = tb_path

        return paths

    # ─────────────────────────────────────────────────────────────────────────
    # Estatisticas
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        pruned    = getattr(self, '_pruned_count', 0)
        n_not     = sum(1 for g in self._gates if g.is_not)
        n_nand    = len(self._gates) - n_not
        total_raw = len(self._gates) + pruned
        cr        = self._concentration_report

        return {
            'total_raw':         total_raw,
            'total_gates':       len(self._gates),
            'nand_gates':        n_nand,
            'not_gates':         n_not,
            'pruned_dead':       pruned,
            'depth':             len(self.model.layers),
            'n_inputs':          self.n_in,
            'n_outputs':         self.n_out,
            'module_name':       self.module_name,
            'gamma_synthesis':   self.gamma_synthesis,
            'min_concentration': cr['min_concentration'],
            'synthesis_safe':    len(cr['warnings']) == 0,
        }
