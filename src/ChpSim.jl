module ChpSim

using Random

import Base.show

export ChpState, cnot!, hadamard!, phase!, measure!


"""
    ChpState(num_qubits[, bitpack=true])

The state of a quantum stabilizer circuit simulation.  I.e. the simulated state
of a quantum computer after applying only Clifford operations to qubits that all
start in the `|0⟩` state.

Supported operations: `cnot!`, `hadamard!`, `phase!`, `measure!`
"""
struct ChpState{MatrixT<:AbstractMatrix{Bool}, VectorT<:AbstractVector{Bool}}
    n::Int
    x::MatrixT
    z::MatrixT
    r::VectorT
    _tmp::VectorT
    _tmp_s::VectorT

    function ChpState(num_qubits, x, z, r)
        @assert size(x) == (2num_qubits, num_qubits)
        @assert size(z) == (2num_qubits, num_qubits)
        @assert size(r) == (2num_qubits,)
        tmp = similar(r)
        tmp_s = similar(r, 1)
        new{typeof(x), typeof(r)}(num_qubits, x, z, r, tmp, tmp_s)
    end
end

function ChpState(MatrixT::DataType, VectorT::DataType, num_qubits::Integer)
    x = similar(MatrixT, 2num_qubits, num_qubits)
    z = similar(MatrixT, 2num_qubits, num_qubits)
    r = similar(VectorT, 2num_qubits)
    x .= 0
    x[CartesianIndex.(1:num_qubits, 1:num_qubits)] .= 1  # Primary diagonal
    z .= 0
    # Diagonal offset down by n
    z[CartesianIndex.(num_qubits+1:2num_qubits, 1:num_qubits)] .= 1
    r .= 0
    ChpState(num_qubits, x, z, r)
end

function ChpState(num_qubits::Integer; bitpack::Bool=false)
    ChpState(bitpack ? BitMatrix : Matrix{Bool},
             bitpack ? BitVector : Vector{Bool},
             num_qubits)
end

function Base.show(io::IO, self::ChpState)
    write(io, "$(typeof(self))($(self.n), $(self.x), $(self.z), $(self.r))")
end

function Base.show(io::IO, ::MIME"text/plain", self::ChpState)
    function _cell(row::Int, col::Int)::String
        k = self.x[row, col] + 2*self.z[row, col]
        [".", "X", "Z", "Y"][k+1]
    end
    function _row(row::Int)::String
        result = self.r[row] ? "-" : "+"
        result *= join(_cell(row, col) for col in 1:self.n)
    end
    z_obs = (_row(row) for row in 1:self.n)
    sep = '-' ^ (self.n + 1)
    x_obs = (_row(row) for row in self.n+1:2*self.n)
    write(io, join((z_obs..., sep, x_obs...), "\n"))
end


"""
    MeasureResult(value, determined)

The result of a qubit measurement returned by `measure!`.
"""
struct MeasureResult
    value::Bool
    determined::Bool

    function MeasureResult(value; determined)
        new(value, determined)
    end
end

function Base.show(io::IO, self::MeasureResult)
    write(io, "$(typeof(self))($(Int(self.value)), "
              * "determined=$(self.determined))")
end

function Base.show(io::IO, ::MIME"text/plain", self::MeasureResult)
    write(io, "$(Int(self.value)) "
              * "($(self.determined ? "determined" : "random"))")
end

MeasureResult(1, determined=false)


"""
    cnot!(state, control, target)

Perform a CNOT gate and update the state.
"""
function cnot!(state::ChpState, control::Int, target::Int)::Nothing
    @assert control != target
    state.r .⊻= ((@view state.x[:, control])
                 .& (@view state.z[:, target])
                 .& ((@view state.x[:, target])
                     .⊻ (@view state.z[:, control])
                     .⊻ true))
    state.x[:, target] .⊻= @view state.x[:, control]
    state.z[:, control] .⊻= @view state.z[:, target]
    nothing
end

"""
    hadamard!(state, qubit)

Perform a Hadamard gate and update the state.
"""
function hadamard!(state::ChpState, qubit::Int)::Nothing
    state.r .⊻= (@view state.x[:, qubit]) .& (@view state.z[:, qubit])
    state._tmp .= @view state.x[:, qubit]
    state.x[:, qubit] .= @view state.z[:, qubit]
    state.z[:, qubit] .= state._tmp
    nothing
end

"""
    phase!(state, qubit)

Perform a phase gate (S gate) and update the state.
"""
function phase!(state::ChpState, qubit::Int)::Nothing
    state.r .⊻= (@view state.x[:, qubit]) .& (@view state.z[:, qubit])
    state.z[:, qubit] .⊻= @view state.x[:, qubit]
    nothing
end

"""
    measure!(state, qubit; rng=GLOBAL_RNG, bias=0.5)

Perform a measurement in the Z basis and return the result.
If the qubit was in superposition, pick a random result and update the state.

# Arguments
- `qubit::Int`: The qubit index to measure.
- `rng::AbstractRNG`: The random number generator to use.
- `bias::Real`: The artificial probablility of measuring a 1 randomly.
"""
function measure!(state::ChpState, qubit::Int;
                  rng::AbstractRNG=Random.GLOBAL_RNG, bias::Real=0.5
                 )::MeasureResult
    for p in 1:state.n
        if state.x[state.n+p, qubit]
            return _measure_random(state, qubit, p, rng, bias)
        end
    end
    _measure_determined(state, qubit)
end

function _measure_determined(state::ChpState, qubit::Int)::MeasureResult
    state._tmp .= 0
    state._tmp_s .= 0
    tx = @view state._tmp[1:state.n]
    tz = @view state._tmp[state.n+1:2state.n]
    tr = @view state._tmp_s[]
    for i in 1:state.n
        if state.x[i, qubit]
            _row_mult(tx, tz, state._tmp_s,
                      (@view state.x[i + state.n, :]),
                      (@view state.z[i + state.n, :]),
                      (@view state.r[i + state.n]))
        end
    end
    MeasureResult(tr[], determined=true)
end

function _measure_random(state::ChpState, qubit::Int, p::Int, rng::AbstractRNG,
                         bias::Real)::MeasureResult
    @assert state.x[state.n + p, qubit]
    state.x[p, :] .= @view state.x[state.n + p, :]
    state.x[state.n + p, :] .= 0
    state.z[p, :] .= @view state.z[state.n + p, :]
    state.z[state.n + p, :] .= 0
    state.r[p] = state.r[state.n + p]
    state.r[state.n + p] = 0
    state.z[state.n + p, qubit] = 1
    state.r[state.n + p] = rand(rng) < bias

    px = @view state.x[p, :]
    pz = @view state.z[p, :]
    pr = @view state.r[p]
    for i in 1:2state.n
        if state.x[i, qubit] && i != p && i != state.n + p
            _row_mult((@view state.x[i, :]),
                      (@view state.z[i, :]),
                      (@view state.r[i]),
                      px, pz, pr)
        end
    end
    MeasureResult(state.r[state.n + p], determined=false)
end

function _row_mult(xrow1, zrow1, rrow1, xrow2, zrow2, rrow2)
    rrow1[] = _row_product_sign(xrow1, zrow1, rrow1, xrow2, zrow2, rrow2)
    xrow1 .⊻= xrow2
    zrow1 .⊻= zrow2
end

function _row_product_sign(xrow1, zrow1, rrow1, xrow2, zrow2, rrow2)
    pauli_phases = sum(
        _pauli_product_phase(xrow1[j], zrow1[j], xrow2[j], zrow2[j])
        for j in 1:length(xrow1)
    )
    @assert pauli_phases & 1 == 0
    p = Bool((pauli_phases >> 1) & 1)
    rrow1[] ⊻ rrow2[] ⊻ p
end

function _pauli_product_phase(x1::Bool, z1::Bool, x2::Bool, z2::Bool)::Int
    ixzy1 = x1 | z1 << 1
    ixzy2 = x2 | z2 << 1
    # If either pauli is I or both are the same
    (ixzy1 == 0 || ixzy2 == 0) && return 0
    # +1 if p1->p2 is increasing X->Y, Y->Z, or Z->X
    # -1 if decending
    # 0 if p1 == p2
    return mod(ixzy1 - ixzy2 + 1, 3) - 1
end


end
