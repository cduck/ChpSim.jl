# ChpSim

A simple simulator for quantum stabilizer circuits based on
[Scott Aaronson, Daniel Gottesman, *Improved Simulation of Stabilizer Circuits*](https://arxiv.org/abs/quant-ph/0406196)
for [Julia](https://julialang.org/).
Adapted from [a Python implementation](https://github.com/Strilanc/python-chp-stabilizer-simulator)
and uses a similar API.

This simulator can efficiently simulate Clifford operations on many qubits but does not support non-Clifford operations.


# Install

Install ChpSim with Julia's package manager:

```bash
julia -e 'using Pkg; Pkg.add("ChpSim")'
```


# Examples

```julia
using ChpSim
sim = ChpState(6)

hadamard!(sim, 1)     # Hadamard gate on the first qubit
phase!(sim, 6)        # S gate on the last qubit
cnot!(sim, 2, 3)      # CNOT gate with control on qubit 2 and target on 3
r = measure!(sim, 3)  # Measure qubit 3

r.value       # The boolean measurement, true or false
r.determined  # False if the qubit was in superposition before measurement
```

```julia
# Uses 1/8 the memory but is typically slower
using ChpSim
sim = ChpState(6, bitpack=true)
```
