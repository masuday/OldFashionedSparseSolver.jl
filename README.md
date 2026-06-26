# OldFashionedSparseSolver

This package implements a simple sparse solver for a symmetric, positive definite system, as described by Stewart (2003).
Sparse inversion, also known as selected inversion, using the Takahashi method (Erisman and Tinney, 1973), is also implemented with a modification to support positive semi-definite matrices in least-squares problems (Boldman et al., 1995).

The purpose of this package is to demonstrate sparse techniques with simple code.
Although the code is still useful in some domains, other libraries, such as `SuiteSparse` included with Julia, are generally faster and more reliable for sparse-matrix computations.
Please use such libraries when appropriate.

## Example

First, prepare a left-hand side matrix and a right-hand side vector.

```julia
julia> using LinearAlgebra
julia> using SparseArrays
julia> n = 10
julia> x = ones(n,1)
julia> A = sparse(Symmetric( -1.0*(triu(rand(n,n),1).>0.6) + 1.0*n*Matrix(I,n,n) ))
julia> b = A*x
```

The actual operations are as follows.

```julia
julia> using AMD
julia> using OldFashionedSparseSolver
julia> perm = amd(A)
julia> L = nzstruct(A,perm)
julia> transfer!(A,L)
julia> ldlt!(L)
julia> sol = L \ b
julia> spinv!(L, A)
```

The sequence for solving a sparse system is as follows.

1. Prepare a permutation vector, which can be obtained with other packages such as AMD or Metis.
2. Perform the symbolic factorization using `nzstruct`, which returns an object for a factor (`L` above).
3. Transfer the original matrix `A` to the factor object `L`.
4. Perform the numerical (LDLT) factorization using `ldlt!`. The factor object will be updated.
5. Solve the equations using the factor with `L \ b` or `spsolve(L,b)`.
6. Optional: the function `spinv!` computes the sparse inverse. Both the factor object and the original matrix will be updated.

You can replace steps 1-4 with `L = spchol(A)`. See below for details.

## Functions

In the following explanation, `A` is a sparse, symmetric, positive-definite matrix, `L` is a factor object, `b` is a single-right-hand-side vector, `B` is a multiple-right-hand-side matrix, and `x` and `X` are a solution vector and matrix, respectively.

### Utility routines

* `transfer!(A,L)` - Transfer the elements in the original matrix `A` to an object `L`.
* `transfer!(L,A)` - Transfer the elements in an object `L` to a sparse matrix `A`.
* `zero!(L)` - Clear all nonzero values in `L`, while leaving the nonzero structure unchanged. In other words, return to the state after symbolic factorization.
* `F = Matrix(L)` - Convert `L` to a full (dense) matrix `F`.
* `S = sparse(L)` - Convert `L` to a sparse matrix `S`.
* `d = diag(L)` - Extract the diagonal elements of `L` and return a vector `d`.
* `d,T = getfactor(L)` - Return a vector `d` for the diagonal elements and a matrix `T` for the off-diagonal elements of `L`.
* `spshow(L [,showfull=false])` - Show the nonzero structure in the lower triangular part of `L`. Show all elements if `showfull=true`.
* `spshow(A [,showfull=true])` - Same as above, but for a sparse matrix.
* `n = rank(L,tol)` - Count the number of diagonal elements greater than `tol` (default = 1e-9).

### Computational routines

* `L = nzstruct(A,perm)` - Perform the symbolic factorization of `A` with a permutation vector `perm` and return `L`.
* `ldlt!(L)` - Perform the numerical (LDLT) factorization and update `L` with the factor.
* `L = spchol(A)` - A batch function that performs `perm=amd(A); L=nzstruct(A,perm); transfer!(A,L); ldlt!(L)` sequentially.
* `L = spchol(A,perm)` - Same as above, but using the user-supplied permutation vector `perm`.
* `x = spsolve(L,b)` - Solve the equations using the factor.
* `X = spsolve(L,B)` - Same as above.
* `x = L \ b` and `X = L \ B` - Alternative ways to solve the equations.
* `spinv!(L)` - Compute the sparse inverse. The object `L` will be updated.
* `spinv!(L,A)` - Compute the sparse inverse of `A`. The object `L` will be updated, and the original matrix `A` will also be rewritten with the elements.

## Note

This module accepts semi-definite matrices arising from least-squares problems.
Once the program finds a zero pivot during factorization, the entire column, including the diagonal element, in the factor will be filled with zero (Boldman et al., 1995).
The resulting inverse will be a generalized inverse.
This *ad hoc* method may make the computations unstable when solving sparse systems or computing sparse inverses.
Please use the sparse QR method in `SuiteSparse` (Julia built-in) if you need a more stable result.

In all cases, the input matrix `A` must contain all diagonal elements, even if one of them is exactly 0.
The function `nzstruct` stops with an error if `A` has no diagonal entry.

This package is not multi-threaded, and it would be slow for a large matrix with many nonzero entries.
Please use the sparse Cholesky method in `SuiteSparse` with multi-threading if the matrix is positive definite.

## References

* Boldman, K. G., L. A. Kriese, L. D. Van Vleck, C. P. Van Tassell, and S. D. Kachman. 1995. A manual for use of MTDFREML. A set of programs to obtain estimates of variances and covariances (DRAFT). U.S. Department of Agriculture, Agricultural Research Service.
* Stewart, G. W. 2003. Building an old-fashioned sparse solver. Technical Report. UMIACS-TR-2003-95. University of Maryland.
* Erisman, A. M. and W. F. Tinney. 1973. On computing certain elements of the inverse of a sparse matrix. Communications of the ACM. 18:177-179.
