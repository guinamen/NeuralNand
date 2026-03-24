# NeuralNand

### Differentiable NAND Networks for Direct Digital Circuit Synthesis

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-brightgreen)
![Status](https://img.shields.io/badge/status-research-orange)
![Hardware](https://img.shields.io/badge/target-Verilog%20synthesis-lightgrey)
![ML](https://img.shields.io/badge/method-differentiable%20logic-purple)

NeuralNand is a **differentiable neural architecture for synthesizing
digital circuits directly from truth tables** using **only NAND gates**.

The network is trained using gradient descent and then **collapsed into
a hardware‑synthesizable Verilog netlist**, preserving a **1:1 mapping
between neurons and NAND gates**.

This project explores the intersection of:

-   Differentiable Logic Networks
-   Neural Architecture Search
-   Digital Logic Synthesis
-   Neuro‑symbolic computing

------------------------------------------------------------------------

# Project Goal

Train a neural network composed exclusively of NAND neurons and convert
it into a **real digital circuit**.

The network learns logic structure during training and eventually
**crystallizes into a discrete Boolean circuit**.

------------------------------------------------------------------------

# High‑Level Architecture

``` mermaid
flowchart LR

A[Arithmetic Operator] --> B[Truth Table Generator]
B --> C[Training Dataset]
C --> D[NAND Neural Network]
D --> E[Annealing + Gradient Training]
E --> F[Crisp Circuit Collapse]
F --> G[Verilog Netlist]
G --> H[Digital Circuit]
```

------------------------------------------------------------------------

# NAND Neuron

Each neuron is a **differentiable relaxation of a NAND gate**.

Forward equation:

f(a,b,T) = 1 − σ(T (w₁a + w₂b − 1.5))

Where

-   σ is the sigmoid
-   T is temperature controlling gate hardness

As **T → ∞**, the neuron becomes an exact NAND gate.

------------------------------------------------------------------------

# Network Structure

``` mermaid
graph TD

I1(Input bits)
I2(Input bits)

L1[Layer 1 NANDs]
L2[Layer 2 NANDs]
L3[Layer 3 NANDs]

O[Output bits]

I1 --> L1
I2 --> L1
L1 --> L2
L2 --> L3
L3 --> O
```

Properties:

-   Fixed arity: **2 inputs per neuron**
-   Direct mapping to hardware
-   DAG topology
-   Skip connections to inputs

------------------------------------------------------------------------

# Training Dynamics

Training uses **annealing** to gradually transform soft logic into
discrete gates.

``` mermaid
graph LR

A[Soft Logic] --> B[Structure Formation]
B --> C[Hard Logic]
C --> D[Discrete Circuit]
```

A single scalar **γ** controls:

-   Gate temperature
-   Routing hardness
-   Loss weighting
-   Regularization

------------------------------------------------------------------------

# Loss Function

Total loss:

L = α L_wbce + β L_arith + λ L_reg

Components:

### Weighted BCE

Bit‑level accuracy with MSB emphasis.

### Arithmetic Loss

Measures numeric error of the integer result.

### Regularization

Encourages NAND symmetry:

(w₁ − 1)² + (w₂ − 1)²

------------------------------------------------------------------------

# Dataset Generation

Instead of datasets, the system generates **complete truth tables**.

  Operator   Inputs   Outputs
  ---------- -------- ---------
  add        2n       n+1
  sub        2n       n+1
  mul        2n       2n
  cmp        2n       1
  max        2n       n
  min        2n       n

Example: 4‑bit addition

2\^(2n) = 256 training rows.

------------------------------------------------------------------------

# Example Training Pipeline

``` mermaid
flowchart TD

A[generate(op,n)]
B[dataset X,Y]
C[NANDNet]
D[trainer]
E[annealing]
F[circuit collapse]
G[verilog]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
```

------------------------------------------------------------------------

# Example Usage

``` python
from neuralnand import train

model = train(
    operator="add",
    bits=4
)

model.export_verilog("adder4.v")
```

Generated circuit example:

``` verilog
nand g0 (w0, a0, b0);
nand g1 (w1, w0, a1);
nand g2 (sum0, w1, carry0);
```

------------------------------------------------------------------------

# Research Motivation

NeuralNand investigates whether neural training can act as a **circuit
discovery mechanism**.

Instead of designing circuits manually, the network **learns Boolean
structure from data**.

Research questions include:

-   Can NAND‑only networks learn arbitrary Boolean functions?
-   How optimal are the resulting circuits?
-   Can approximate circuits emerge naturally?
-   How does circuit complexity scale with bit width?

------------------------------------------------------------------------

# Key Differences From Standard Neural Networks

  Property           NeuralNand         Standard NN
  ------------------ ------------------ --------------------
  Activation         NAND relaxation    ReLU / sigmoid
  Output             Boolean circuit    Numeric prediction
  Interpretability   Fully structural   Limited
  Deployment         Verilog hardware   CPU/GPU inference

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/guinamen/NeuralNand
cd NeuralNand
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Roadmap

Planned improvements:

-   Hard Straight‑Through training
-   Gumbel routing
-   Automatic circuit simplification
-   Hardware cost optimization
-   FPGA benchmarking

------------------------------------------------------------------------

# Vision

NeuralNand explores a broader concept:

> Neural networks can function as **program synthesizers** that generate
> symbolic computational artifacts.

In this project, the artifact is a **digital circuit**.

------------------------------------------------------------------------

# License

MIT
