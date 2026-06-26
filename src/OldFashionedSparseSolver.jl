module OldFashionedSparseSolver

using AMD

import Base: Matrix
import LinearAlgebra: (\), diag, zero, ldlt!, Symmetric, issymmetric, tril, rank
import SparseArrays: sparse, SparseMatrixCSC

export LDLtFactor
export transfer!, zero!, full, sparse, diag, getfactor, spshow
export nzstruct, ldlt!, spchol, spdet, splogdet, spsolve, spsolve!, spinv!, (\)

"""
Storage for LDLt factorization in the Old Fashioned solver.
"""
mutable struct LDLtFactor{Tv,Ti}
   n::Int
   nnz::Int
   iperm::Vector{Ti}
   ia::Vector{Ti}
   ja::Vector{Ti}
   a::Vector{Tv}
   status::String
end

include("subroutines.jl")
include("batch.jl")

end
