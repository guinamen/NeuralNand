"""
nand_net.py
-----------
Rede neural composta exclusivamente de portas NAND com relaxacao continua.

Especificacao implementada:
  - Neuronio NAND: f(a,b;T) = 1 - sigmoid(T * (w1*a_soft + w2*b_soft - 1.5))
  - Pesos positivos via softplus: w = softplus(rho)  -->  w in (0, +inf)
  - Bias fixo = -1.5  (nao treinavel)
  - Selecao de entradas via logits + softmax(theta / tau)
  - Topologia Opcao B: cada camada acessa anterior + entradas originais (skip)
  - Afunilamento linear de n1 neuronios ate n_outputs
  - Um escalar gamma governa T, tau, alpha, beta e lambda simultaneamente
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Configuracao de annealing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnealingConfig:
    """
    Escalar de controle unico gamma(e) que governa todo o sistema dinamico.

    gamma(e) = gamma_0 + (gamma_max - gamma_0) * (1 - exp(-e / tau_sched))

    Derivados em cada epoca:
        T       = gamma              endurece o neuronio NAND
        tau     = 1 / gamma          endurece a selecao de conexoes
        alpha   = 1 - s(t)           reduz peso BCE bit-a-bit
        beta    = s(t)               aumenta peso erro aritmetico
        lambda  = lmax * t^kappa     aumenta regularizacao dos pesos
    """
    gamma_0:     float = 0.5
    gamma_max:   float = 20.0
    tau_sched:   float = 60.0
    gamma_crisp: float = 15.0
    sharpness:   float = 6.0
    kappa:       float = 2.0
    lambda_max:  float = 1e-3

    def gamma_at(self, epoch: int) -> float:
        return self.gamma_0 + (self.gamma_max - self.gamma_0) * (
            1.0 - math.exp(-epoch / self.tau_sched)
        )

    def t_norm(self, gamma: float) -> float:
        """Progresso normalizado em [0, 1]."""
        return (gamma - self.gamma_0) / (self.gamma_max - self.gamma_0)

    def alpha_beta(self, gamma: float):
        t = self.t_norm(gamma)
        s = 1.0 / (1.0 + math.exp(-self.sharpness * (t - 0.5)))
        return 1.0 - s, s   # alpha, beta  (soma = 1 por construcao)

    def lambda_val(self, gamma: float) -> float:
        t = self.t_norm(gamma)
        return self.lambda_max * (t ** self.kappa)

    # gamma usado na sintese Verilog e em crisp_connections()
    # precisa ser alto o suficiente para forcar softmax(theta/tau) one-hot
    # Com delta de logits tipico de 0.05: gamma >= 138
    # Valor conservador: 200
    gamma_synth: float = 200.0

    def is_crisp(self, gamma: float) -> bool:
        return gamma >= self.gamma_crisp


# ─────────────────────────────────────────────────────────────────────────────
# Neuronio NAND diferenciavel
# ─────────────────────────────────────────────────────────────────────────────

class NANDNeuron(nn.Module):
    """
    Porta NAND com relaxacao continua e selecao diferenciavel de entradas.

    Parametros treinados:
        rho1, rho2  in R         -- pesos via softplus, garantidamente positivos
        theta_a     in R^n_pool  -- logits de selecao da entrada A
        theta_b     in R^n_pool  -- logits de selecao da entrada B

    Forward dado gamma:
        w1, w2   = softplus(rho1), softplus(rho2)
        tau      = 1 / gamma
        alpha    = softmax(theta_a / tau)
        beta     = softmax(theta_b / tau)
        a_soft   = sum_k alpha_k * pool_k
        b_soft   = sum_k beta_k  * pool_k
        y        = 1 - sigmoid(gamma * (w1*a_soft + w2*b_soft - 1.5))

    Limites:
        gamma -> inf  : tau->0 cristaliza conexoes; T->inf binariza saida
        caso i*=j*    : NAND(x,x) = NOT(x) -- aceito, sintetizavel
    """

    BIAS = -1.5  # fixo pela especificacao

    def __init__(self, n_pool: int, neuron_idx: int = 0):
        """
        neuron_idx : indice do neuronio na camada.

        Inicializacao por rotacao ciclica forte:
            i_pref = neuron_idx % n_pool
            j_pref = (neuron_idx + n_pool//2 + 1) % n_pool

            theta_a e inicializado como one-hot suavizado em i_pref
            theta_b e inicializado como one-hot suavizado em j_pref

            Isso garante que neuronio k comece preferindo um par (i,j)
            diferente do neuronio k+1, cobrindo todo o pool ciclicamente.
            O gradiente pode migrar para outro par se necessario.
        """
        super().__init__()
        rho_star = math.log(math.e - 1.0)
        self.rho1 = nn.Parameter(
            torch.tensor(rho_star) + torch.randn(1).squeeze() * 0.01
        )
        self.rho2 = nn.Parameter(
            torch.tensor(rho_star) + torch.randn(1).squeeze() * 0.01
        )
        # One-hot suavizado: posicao preferida recebe 1.0, demais recebem ruido pequeno
        # Com tau_0 = 1/gamma_0 = 2.0, softmax(1.0/2.0) ~ 0.57 vs 0.14 por posicao
        # Diferenca suficiente para guiar sem travar
        i_pref  = neuron_idx % n_pool
        j_pref  = (neuron_idx + max(1, n_pool // 2)) % n_pool
        theta_a = torch.randn(n_pool) * 0.01
        theta_b = torch.randn(n_pool) * 0.01
        theta_a[i_pref] += 1.0
        theta_b[j_pref] += 1.0
        self.theta_a = nn.Parameter(theta_a)
        self.theta_b = nn.Parameter(theta_b)

    def forward(self, pool: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        pool  : (batch, n_pool) em [0,1]
        gamma : escalar de annealing

        Retorna: (batch,)
        """
        tau = 1.0 / gamma
        w1  = F.softplus(self.rho1)
        w2  = F.softplus(self.rho2)

        alpha  = F.softmax(self.theta_a / tau, dim=0)          # (n_pool,)
        beta   = F.softmax(self.theta_b / tau, dim=0)          # (n_pool,)
        a_soft = (pool * alpha.unsqueeze(0)).sum(dim=1)        # (batch,)
        b_soft = (pool * beta.unsqueeze(0)).sum(dim=1)         # (batch,)

        z = w1 * a_soft + w2 * b_soft + self.BIAS
        return 1.0 - torch.sigmoid(gamma * z)

    def crisp_connections(self, gamma_synth: float = 1000.0):
        """
        Indices (i*, j*) para sintese Verilog.

        PROBLEMA com argmax puro: com gamma_crisp=19, tau=1/19~0.053,
        o softmax ainda tem distribuicao residual significativa.
        Outros indices contribuem para a_soft/b_soft e mudam o resultado
        em relacao a simulacao booleana pura com conexao discreta.

        SOLUCAO: usa gamma_synth muito alto (default=1000) para forcar
        tau -> 0 -> softmax one-hot antes de extrair o argmax.
        Com tau=0.001 a concentracao e > 0.9999 em todos os neuronios.
        """
        tau   = 1.0 / gamma_synth
        alpha = F.softmax(self.theta_a / tau, dim=0)
        beta  = F.softmax(self.theta_b / tau, dim=0)
        return alpha.argmax().item(), beta.argmax().item()


    def softmax_concentration(self, tau: float) -> tuple:
        """
        Retorna a concentracao do softmax para ambos os slots (a, b).
        Concentracao = probabilidade do indice dominante.
        Valor 1.0 = completamente one-hot (crisp perfeito).
        Valor 1/n = completamente uniforme (nao aprendeu conexao).

        Usado para verificar se o neuronio esta pronto para sintese.
        Recomendado: concentracao >= 0.99 para sintese segura.
        """
        import torch.nn.functional as F_local
        import torch
        alpha = F_local.softmax(self.theta_a / tau, dim=0)
        beta  = F_local.softmax(self.theta_b / tau, dim=0)
        return alpha.max().item(), beta.max().item()

    def crisp_weights(self):
        """(w1, w2) efetivos."""
        return F.softplus(self.rho1).item(), F.softplus(self.rho2).item()


# ─────────────────────────────────────────────────────────────────────────────
# Camada NAND
# ─────────────────────────────────────────────────────────────────────────────

class NANDLayer(nn.Module):
    """
    N neuronios NAND em paralelo compartilhando o mesmo pool de entradas.
    """

    def __init__(self, n_neurons: int, n_pool: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_pool    = n_pool
        self.neurons   = nn.ModuleList([NANDNeuron(n_pool, i) for i in range(n_neurons)])

    def forward(self, pool: torch.Tensor, gamma: float) -> torch.Tensor:
        """pool: (batch, n_pool)  -->  (batch, n_neurons)"""
        return torch.stack([n(pool, gamma) for n in self.neurons], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Rede NAND completa
# ─────────────────────────────────────────────────────────────────────────────

class NANDNet(nn.Module):
    """
    Rede feedforward composta exclusivamente de portas NAND diferenciáveis.

    Topologia:
        L0 (entradas)
        L1 : n1 neuronios, pool = L0
        L2 : n2 neuronios, pool = L1 + L0  (skip)
        ...
        Lk : nk neuronios, pool = L(k-1) + L0  (skip)
        Ls : n_outputs neuronios  (camada de saida)

    Afunilamento: nl = round(n1 + (n_outputs - n1) * (l-1)/(L-1))
    Invariante DAG: cada camada acessa apenas sinais anteriores -- sem ciclos.

    Args:
        n_inputs  : bits de entrada do circuito
        n1        : neuronios na primeira camada oculta
        n_layers  : numero de camadas ocultas
        n_outputs : bits de saida do circuito

    Exemplo:
        net = NANDNet(n_inputs=6, n1=10, n_layers=3, n_outputs=4)
        y   = net(x, gamma=5.0)
    """

    def __init__(self, n_inputs: int, n1: int, n_layers: int, n_outputs: int):
        super().__init__()
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs

        hidden = self._hidden_sizes(n1, n_outputs, n_layers)
        self.hidden = hidden

        layers = []
        # Primeira camada oculta: pool = apenas entradas
        layers.append(NANDLayer(n_neurons=hidden[0], n_pool=n_inputs))
        # Camadas seguintes: pool = anterior + entradas (skip)
        for i in range(1, len(hidden)):
            layers.append(NANDLayer(n_neurons=hidden[i], n_pool=hidden[i-1] + n_inputs))
        # Camada de saida
        layers.append(NANDLayer(n_neurons=n_outputs, n_pool=hidden[-1] + n_inputs))

        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _hidden_sizes(n1: int, n_out: int, n_layers: int) -> List[int]:
        if n_layers == 1:
            return [max(n_out + 1, n1)]
        sizes = []
        for l in range(n_layers):
            t = l / (n_layers - 1)
            n = round(n1 + (n_out - n1) * t)
            sizes.append(max(n_out + 1, n))
        return sizes

    def forward(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        x     : (batch, n_inputs) em {0,1} ou [0,1]
        gamma : escalar de annealing atual
        """
        h = self.layers[0](x, gamma)
        for layer in self.layers[1:]:
            pool = torch.cat([h, x], dim=1)
            h    = layer(pool, gamma)
        return h

    def count_gates(self) -> int:
        return sum(len(l.neurons) for l in self.layers)

    def summary(self) -> str:
        lines = [
            f'NANDNet  {self.n_inputs} in -> {self.n_outputs} out  '
            f'({self.count_gates()} portas NAND totais)',
            f'{"Camada":<14} {"Neuronios":>10} {"Pool":>8}',
            '-' * 36,
            f'{"L0 (input)":<14} {self.n_inputs:>10} {"--":>8}',
        ]
        for i, layer in enumerate(self.layers):
            tag = 'saida' if i == len(self.layers) - 1 else f'oculta {i+1}'
            lines.append(f'{"L"+str(i+1)+" ("+tag+")":<14} {layer.n_neurons:>10} {layer.n_pool:>8}')
        return '\n'.join(lines)
