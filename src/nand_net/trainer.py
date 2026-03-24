"""
trainer.py
----------
Loop de treino da NANDNet com annealing unificado.

Estrategia de gradiente — curriculum de gamma:
  - Forward e backward usam SEMPRE o mesmo gamma
  - gamma sobe gradualmente via schedule exponencial
  - Nao ha dualidade entre gamma de avaliacao e gamma de treino
  - O gradiente enfraquece naturalmente com gamma, mas isso e correto:
    quando gamma e alto os parametros ja devem estar proximos do otimo

Isso elimina a discrepancia que causava a degradacao ao entrar no regime crisp.

Loss total:
    L = alpha(e) * L_wbce(pesos_adaptativos)  +  beta(e) * L_arith  +  lambda(e) * L_reg

Criterio de parada:
    L_arith@gamma_atual < epsilon^2  E  gamma >= gamma_crisp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, List, Callable

from .nand_net import NANDNet, AnnealingConfig
from .dataset  import DatasetMeta


# ─────────────────────────────────────────────────────────────────────────────
# Configuracao
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    lr:            float = 3e-3
    epochs:        int   = 2500
    epsilon:       float = 0.05
    log_every:     int   = 100
    device:        str   = 'cpu'
    grad_clip:     float = 1.0
    # curvatura da transicao dos pesos adaptativos da loss
    # delta alto: pesos uniformes por mais tempo, divergem so no final
    delta_weights: float = 4.0


@dataclass
class EpochState:
    epoch:       int
    gamma:       float
    alpha:       float
    beta:        float
    lam:         float
    loss_wbce:   float
    loss_arith:  float
    loss_reg:    float
    loss_total:  float
    acc_bit:     float
    acc_row:     float
    is_crisp:    bool


# ─────────────────────────────────────────────────────────────────────────────
# Funcoes de loss
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_weights(
    weights_final: torch.Tensor,
    t: float,
    delta: float = 4.0,
) -> torch.Tensor:
    """
    Interpola entre pesos uniformes (t=0) e pesos finais (t=1).

    w_k(t) = w_k_final * t^delta  +  (1/n) * (1 - t^delta)

    delta alto: pesos permanecem uniformes por mais tempo,
    so divergindo no final quando a rede ja aprendeu os bits basicos.
    """
    n       = weights_final.shape[0]
    td      = t ** delta
    uniform = torch.ones_like(weights_final) / n
    return weights_final * td + uniform * (1.0 - td)


def loss_weighted_bce(
    pred:    torch.Tensor,
    target:  torch.Tensor,
    weights: torch.Tensor,
    t:       float = 1.0,
    delta:   float = 4.0,
    eps:     float = 1e-7,
) -> torch.Tensor:
    """
    BCE com pesos adaptativos por posicao de bit.
    pred, target : (batch, bits_out)
    weights      : (bits_out,)  pesos finais w_k = 2^k / (2^bits_out - 1)
    t            : progresso normalizado [0,1]
    delta        : curvatura da transicao
    """
    w    = adaptive_weights(weights, t, delta)
    pred = pred.clamp(eps, 1 - eps)
    bce  = -(target * pred.log() + (1 - target) * (1 - pred).log())
    return (bce * w.unsqueeze(0)).mean()


def loss_arithmetic(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Erro aritmetico normalizado (LSB-first)."""
    bits_out = pred.shape[1]
    powers   = torch.pow(2, torch.arange(bits_out, dtype=torch.float32)).to(pred.device)
    max_val  = float((1 << bits_out) - 1)
    pred_int   = (pred   * powers).sum(dim=1)
    target_int = (target * powers).sum(dim=1)
    return ((pred_int - target_int) / max_val).pow(2).mean()


def loss_regularization(model: NANDNet) -> torch.Tensor:
    """L2 dos pesos em torno de w=1 (simetria NAND)."""
    terms = []
    for layer in model.layers:
        for neuron in layer.neurons:
            w1 = F.softplus(neuron.rho1)
            w2 = F.softplus(neuron.rho2)
            terms.append((w1 - 1.0).pow(2) + (w2 - 1.0).pow(2))
    return sum(terms) if terms else torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NANDTrainer:
    """
    Treina a NANDNet com curriculum de gamma.

    Forward e backward usam sempre o mesmo gamma.
    Gamma sobe gradualmente — a rede adapta seus parametros
    continuamente ao endurecimento progressivo da sigmoid,
    sem saltos que destroem a representacao aprendida.

    Salva o melhor estado pelo L_arith durante toda a fase crisp.
    """

    def __init__(
        self,
        model:     NANDNet,
        meta:      DatasetMeta,
        annealing: AnnealingConfig,
        config:    TrainerConfig,
        on_log:    Optional[Callable[[EpochState], None]] = None,
    ):
        self.model     = model.to(config.device)
        self.meta      = meta
        self.annealing = annealing
        self.config    = config
        self.on_log    = on_log or self._default_log
        self.history:  List[EpochState] = []
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> List[EpochState]:
        X = X.to(self.config.device)
        Y = Y.to(self.config.device)
        W = self.meta.weights.to(self.config.device)

        best_arith      = float('inf')
        best_state_dict = None

        for epoch in range(self.config.epochs):
            gamma       = self.annealing.gamma_at(epoch)
            alpha, beta = self.annealing.alpha_beta(gamma)
            lam         = self.annealing.lambda_val(gamma)
            is_crisp    = self.annealing.is_crisp(gamma)
            t_prog      = self.annealing.t_norm(gamma)

            # ── Forward + backward com mesmo gamma ─────────────────────
            self.model.train()
            self.optimizer.zero_grad()

            pred    = self.model(X, gamma)
            l_wbce  = loss_weighted_bce(pred, Y, W,
                                        t=t_prog,
                                        delta=self.config.delta_weights)
            l_arith = loss_arithmetic(pred, Y)
            l_reg   = loss_regularization(self.model)
            loss    = alpha * l_wbce + beta * l_arith + lam * l_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            # ── Metricas ───────────────────────────────────────────────
            with torch.no_grad():
                pred_bin = (pred >= 0.5).float()
                acc_bit  = (pred_bin == Y).float().mean().item()
                acc_row  = (pred_bin == Y).all(dim=1).float().mean().item()

            state = EpochState(
                epoch      = epoch,
                gamma      = gamma,
                alpha      = alpha,
                beta       = beta,
                lam        = lam,
                loss_wbce  = l_wbce.item(),
                loss_arith = l_arith.item(),
                loss_reg   = l_reg.item(),
                loss_total = loss.item(),
                acc_bit    = acc_bit,
                acc_row    = acc_row,
                is_crisp   = is_crisp,
            )
            self.history.append(state)

            # Salva melhor estado na fase crisp
            if is_crisp and l_arith.item() < best_arith:
                best_arith      = l_arith.item()
                best_state_dict = {k: v.clone()
                                   for k, v in self.model.state_dict().items()}

            should_log = (epoch % self.config.log_every == 0) or \
                         (is_crisp and epoch % 20 == 0)
            if should_log:
                self.on_log(state)

            # ── Early stopping ─────────────────────────────────────────
            if is_crisp and l_arith.item() < self.config.epsilon ** 2:
                self.on_log(state)
                print(f'\n>> Convergiu epoca {epoch} '
                      f'L_arith={l_arith.item():.6f} < {self.config.epsilon**2:.6f}')
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f'\n>> Restaurado melhor estado crisp: L_arith={best_arith:.6f}')

        return self.history

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor,
                 gamma_eval: Optional[float] = None) -> dict:
        """
        Avalia o circuito em modo crisp.

        gamma_eval: gamma usado na avaliacao.
            None  -> usa gamma_crisp (recomendado: modelo foi validado neste valor)
            float -> usa o valor fornecido

        Nota: gamma_max pode quebrar neurônios que convergiram em gamma_crisp
        pois a sigmoid com gamma_max e mais abrupta — neurônios na fronteira
        de decisao podem mudar de lado. Use gamma_crisp para avaliacao correta.
        """
        gamma = gamma_eval if gamma_eval is not None else self.annealing.gamma_crisp
        X, Y  = X.to(self.config.device), Y.to(self.config.device)
        self.model.eval()

        pred     = self.model(X, gamma)
        pred_bin = (pred >= 0.5).float()

        correct_rows = (pred_bin == Y).all(dim=1)
        acc_row      = correct_rows.float().mean().item()
        l_arith      = loss_arithmetic(pred, Y).item()

        errors = []
        for i in range(len(X)):
            if not correct_rows[i]:
                errors.append({
                    'input':    X[i].tolist(),
                    'expected': Y[i].tolist(),
                    'got':      pred_bin[i].tolist(),
                })

        return {
            'gamma_eval': gamma,
            'gamma_crisp': self.annealing.gamma_crisp,
            'acc_row':    acc_row,
            'acc_bit':    (pred_bin == Y).float().mean().item(),
            'loss_arith': l_arith,
            'n_correct':  int(correct_rows.sum()),
            'n_total':    len(X),
            'is_perfect': acc_row == 1.0,
            'errors':     errors,
        }

    @staticmethod
    def _default_log(s: EpochState):
        crisp_tag = '[CRISP]' if s.is_crisp else ''
        print(
            f'e={s.epoch:4d} | g={s.gamma:5.2f} {crisp_tag:<8} | '
            f'a={s.alpha:.2f} b={s.beta:.2f} | '
            f'arith={s.loss_arith:.5f} | acc_row={s.acc_row:.3f}'
        )
