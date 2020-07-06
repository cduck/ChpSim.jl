using Test
import IterTools

using ChpSim


@testset "ChpSim.jl" begin


@testset "convert_result" begin
    @test convert(Bool, ChpSim.MeasureResult(true, determined=false)) == true
    @test convert(Complex, ChpSim.MeasureResult(true, determined=false)) ==
        Complex(true, false)
end

@testset "identity" begin
    s = ChpState(1)
    @test measure!(s, 1) == ChpSim.MeasureResult(false, determined=true)
end

@testset "bit_packed_table" begin
    s = ChpState(2)
    hadamard!(s, 1)
    phase!(s, 1)
    phase!(s, 1)
    hadamard!(s, 1)
    cnot!(s, 1, 2)
    @test measure!(s, 1) == ChpSim.MeasureResult(true, determined=true)
    @test measure!(s, 2) == ChpSim.MeasureResult(true, determined=true)
end

@testset "bit_flip" begin
    s = ChpState(1)
    hadamard!(s, 1)
    phase!(s, 1)
    phase!(s, 1)
    hadamard!(s, 1)
    @test measure!(s, 1) == ChpSim.MeasureResult(true, determined=true)
end

@testset "identity_2" begin
    s = ChpState(2)
    @test measure!(s, 1) == ChpSim.MeasureResult(false, determined=true)
    @test measure!(s, 2) == ChpSim.MeasureResult(false, determined=true)
end

@testset "bit_flip_2" begin
    s = ChpState(2)
    hadamard!(s, 1)
    phase!(s, 1)
    phase!(s, 1)
    hadamard!(s, 1)
    @test measure!(s, 1) == ChpSim.MeasureResult(true, determined=true)
    @test measure!(s, 2) == ChpSim.MeasureResult(false, determined=true)
end

@testset "epr" begin
    s = ChpState(2)
    hadamard!(s, 1)
    cnot!(s, 1, 2)
    v1 = measure!(s, 1)
    v2 = measure!(s, 2)
    @test !v1.determined
    @test v2.determined
    @test v1.value == v2.value
end

@testset "_pauli_product_phase" begin
    paulis = Bool[0 0  # I
                  1 0  # X
                  1 1  # Y
                  0 1] # Z
    expected = [0  0  0  0
                0  0  1 -1
                0 -1  0  1
                0  1 -1  0]
    @test ChpSim._pauli_product_phase(true, false, true, true) == 1
    for i in 1:4, j in 1:4
        @test (ChpSim._pauli_product_phase(paulis[i, :]..., paulis[j, :]...)
               == expected[i, j])
    end
end

@testset "phase_kickback_consume_s_state" begin
    s = ChpState(2)
    hadamard!(s, 2)
    phase!(s, 2)
    hadamard!(s, 1)
    cnot!(s, 1, 2)
    v1 = measure!(s, 2)
    @test !v1.determined
    if v1.value
        phase!(s, 1)
        phase!(s, 1)
    end
    phase!(s, 1)
    hadamard!(s, 1)
    @test measure!(s, 1) == ChpSim.MeasureResult(true, determined=true)
end

@testset "phase_kickback_preserve_s_state" begin
    s = ChpState(2)

    hadamard!(s, 2)
    phase!(s, 2)

    hadamard!(s, 1)

    cnot!(s, 1, 2)
    hadamard!(s, 2)
    cnot!(s, 1, 2)
    hadamard!(s, 2)

    phase!(s, 1)
    hadamard!(s, 1)
    @test measure!(s, 1) == ChpSim.MeasureResult(true, determined=true)
    phase!(s, 2)
    hadamard!(s, 2)
    @test measure!(s, 2) == ChpSim.MeasureResult(true, determined=true)
end

@testset "kickback_vs_stabilizer" begin
    s = ChpState(3)
    hadamard!(s, 3)
    cnot!(s, 3, 1)
    cnot!(s, 3, 2)
    phase!(s, 1)
    phase!(s, 2)
    hadamard!(s, 1)
    hadamard!(s, 2)
    hadamard!(s, 3)
    @test repr(MIME("text/plain"), s) == """
        -Y..
        -.Y.
        +..X
        ----
        +X.X
        +.XX
        +YYZ"""
    v1 = measure!(s, 1, bias=0)
    @test repr(MIME("text/plain"), s) == """
        +X.X
        -.Y.
        +..X
        ----
        +Z..
        +.XX
        +ZYY"""
    v2 = measure!(s, 2, bias=0)
    @test repr(MIME("text/plain"), s) == """
        +X.X
        +.XX
        +..X
        ----
        +Z..
        +.Z.
        -ZZZ"""
    v3 = measure!(s, 3, bias=0)
    @test repr(MIME("text/plain"), s) == """
        +X.X
        +.XX
        +..X
        ----
        +Z..
        +.Z.
        -ZZZ"""
    @test v1 == ChpSim.MeasureResult(false, determined=false)
    @test v2 == ChpSim.MeasureResult(false, determined=false)
    @test v3 == ChpSim.MeasureResult(true, determined=true)
end

@testset "s_state_distillation_low_depth" begin
    for _ in 1:100
        s = ChpState(9)

        stabilizers = [
            (1, 2, 3, 4),
            (1, 2, 5, 6),
            (1, 3, 5, 7),
            (2, 3, 5, 8),
        ]
        checks = [
            (s=[1], q=stabilizers[1]),
            (s=[2], q=stabilizers[2]),
            (s=[3], q=stabilizers[3]),
        ]

        stabilizer_measurements = Bool[]
        anc = 9
        for stab in stabilizers
            hadamard!(s, anc)
            for k in stab
                cnot!(s, anc, k)
            end
            hadamard!(s, anc)
            v = measure!(s, anc)
            @test !v.determined
            if v.value
                hadamard!(s, anc)
                phase!(s, anc)
                phase!(s, anc)
                hadamard!(s, anc)
            end
            push!(stabilizer_measurements, v.value)
        end

        qubit_measurements = Bool[]
        for k in 1:7
            phase!(s, k)
            hadamard!(s, k)
            push!(qubit_measurements, measure!(s, k).value)
        end

        if (sum(stabilizer_measurements) + sum(qubit_measurements)) & 1 == 1
            phase!(s, 8)
            phase!(s, 8)
        end

        phase!(s, 8)
        hadamard!(s, 8)
        r = measure!(s, 8)

        @test r == ChpSim.MeasureResult(false, determined=true)
        for c in checks
            rvs = [stabilizer_measurements[k] for k in c.s]
            rms = [qubit_measurements[k] for k in c.q]
            @test (sum(rvs) + sum(rms)) & 1 == 0
        end
    end
end

@testset "s_state_distillation_low_space" begin
    for _ in 1:100
        s = ChpState(5)

        phasors = [
            (1,),
            (2,),
            (3,),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
        ]

        anc = 5
        for phasor in phasors
            hadamard!(s, anc)
            for k in phasor
                cnot!(s, anc, k)
            end
            hadamard!(s, anc)
            phase!(s, anc)
            hadamard!(s, anc)
            v = measure!(s, anc)
            @test !v.determined
            if v.value
                for k in (phasor..., anc)
                    hadamard!(s, k)
                    phase!(s, k)
                    phase!(s, k)
                    hadamard!(s, k)
                end
            end
        end

        for k in 1:3
            @test measure!(s, k) == ChpSim.MeasureResult(false, determined=true)
        end
        phase!(s, 4)
        hadamard!(s, 4)
        @test measure!(s, 4) == ChpSim.MeasureResult(true, determined=true)
    end
end

@testset "count_s_state_distillation_failure_cases" begin
    distill(errors)::String = distill(Set(errors))
    function distill(errors::Set{Int})::String
        s = ChpState(5)

        phasors = [
            (1,),
            (2,),
            (3,),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
        ]

        anc = 5
        for (e, phasor) in enumerate(phasors)
            hadamard!(s, anc)
            for k in phasor
                cnot!(s, anc, k)
            end
            hadamard!(s, anc)
            phase!(s, anc)

            if e in errors
                phase!(s, anc)
                phase!(s, anc)
            end

            hadamard!(s, anc)
            v = measure!(s, anc)
            @test !v.determined
            if v.value
                hadamard!(s, anc)
                phase!(s, anc)
                phase!(s, anc)
                hadamard!(s, anc)
                for k in phasor
                    hadamard!(s, k)
                    phase!(s, k)
                    phase!(s, k)
                    hadamard!(s, k)
                end
            end
        end

        phase!(s, 4)
        phase!(s, 4)
        phase!(s, 4)
        hadamard!(s, 4)
        v = measure!(s, 4)
        @test v.determined
        hadamard!(s, 4)
        phase!(s, 4)
        checks = [measure!(s, k) for k in 1:3]
        @test all(e.determined for e in checks)
        good_result = !v.value
        checks_passed = !any(e.value for e in checks)
        return if checks_passed
            good_result ? "good" : "ERROR"
        else
            good_result ? "victim" : "caught"
        end
    end

    function classify(errs)::Dict{String, Int}
        result::Dict{String, Int} = Dict()
        for err in errs
            r = distill(err)
            result[r] = get(result, r, 0) + 1
        end
        result
    end

    nones = IterTools.subsets(1:7, 0)
    singles = IterTools.subsets(1:7, 1)
    doubles = IterTools.subsets(1:7, 2)
    triples = IterTools.subsets(1:7, 3)

    @test classify(nones) == Dict("good"=>1)
    @test classify(singles) == Dict("caught"=>3, "victim"=>4)
    @test classify(doubles) == Dict("caught"=>12, "victim"=>9)
    @test classify(triples) == Dict("caught"=>12, "victim"=>16, "ERROR"=>7)
end


end
