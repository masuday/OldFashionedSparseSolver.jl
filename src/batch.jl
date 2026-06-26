
function spshow(L::LDLtFactor; showfull::Bool=false)
   if L.status == ""
      throw(ArgumentError("Nonzero structure undetermined"))
   end
   show_nonzero_structure(L.n, L.ia, L.ja, L.a, showfull)
end

function spshow(A::SparseMatrixCSC; showfull::Bool=true)
   if A.n != A.m
      throw(ArgumentError("Unsymmetric matrix"))
   end
   show_nonzero_structure(A.n, A.colptr, A.rowval, A.nzval, showfull)
end

function isokforldltfact(A::SparseMatrixCSC)
   return has_all_diagonals(A.n, A.colptr, A.rowval)
end

"""
    transfer!(A,L)
    transfer!(L,A)

Transfer a sparse matrix `A` to the `LDLt` structure `L`, which has already been created
by `nzstruct`, and vice versa. The user should transfer `A` to `L` before performing
the LDLt factorization.

### Examples
```jldoctest
julia> using AMD

julia> perm = amd(A)

julia> L = nzstruct(A,perm)

julia> transfer!(A,L)

```
"""
function transfer!(A::SparseMatrixCSC,L::LDLtFactor)
   if L.status != "SymbolicFactorization"
      throw(ArgumentError("Matrix not symbolically factored"))
   end
   if L.n != A.n || A.n!=A.m
      throw(DimensionMismatch("Size mismatch between two matrices"))
   end
   load_sparse_to_factor!(A.n, A.colptr, A.rowval, A.nzval,
                          L.iperm, L.ia, L.ja, L.a)
end

function transfer!(As::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}},L::LDLtFactor) where {Tv,Ti}
   if L.status != "SymbolicFactorization"
      throw(ArgumentError("Matrix not symbolically factored"))
   end
   if L.n != As.data.n || As.data.n!=As.data.m
      throw(DimensionMismatch("Size mismatch between two matrices"))
   end
   A = convert(SparseMatrixCSC,As)
   load_sparse_to_factor!(A.n, A.colptr, A.rowval, A.nzval,
                          L.iperm, L.ia, L.ja, L.a)
end

function transfer!(L::LDLtFactor,A::SparseMatrixCSC)
   if L.status == ""
      throw(ArgumentError("Matrix not initialized"))
   end
   if L.n != A.n || A.n!=A.m
      throw(DimensionMismatch("Size mismatch between two matrices"))
   end
   save_factor_to_sparse!(L.n, A.colptr, A.rowval, A.nzval,
                          L.iperm, L.ia, L.ja, L.a)
end

function transfer!(L::LDLtFactor,As::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
   if L.status == ""
      throw(ArgumentError("Matrix not initialized"))
   end
   if L.n != As.data.n || As.data.n!=As.data.m
      throw(DimensionMismatch("Size mismatch between two matrices"))
   end
   save_factor_to_sparse!(L.n, As, L.iperm, L.ia, L.ja, L.a)
end

function zero!(L::LDLtFactor)
   L.a = zeros(eltype(L.a),L.n)
   L.status = "SymbolicFactorization"
   return nothing
end

function Matrix(L::LDLtFactor; showall::Bool=false)
   A = zeros(eltype(L.a),L.n,L.n)
   for j=1:L.n
      for k=L.ia[j]:L.ia[j+1]-1
         i = L.ja[k]
         A[i,j] = L.a[k]
         if showall
            A[j,i] = A[i,j]
         end
      end
   end
   return A
end

function sparse(L::LDLtFactor)
   if L.status == ""
      throw(ArgumentError("Factor not initialized"))
   end
   return SparseMatrixCSC(L.n,L.n,L.ia,L.ja,L.a)
end

function diag(L::LDLtFactor)
   d = zeros(eltype(L.a),L.n)
   for j=1:L.n
      for k=L.ia[j]:L.ia[j+1]-1
         i = L.ja[k]
         if i==j
            d[i] = L.a[k]
         end
      end
   end
   return d
end

function diag(L::LDLtFactor, perm::Vector{T}) where T
   d = zeros(eltype(L.a),L.n)
   for j=1:L.n
      for k=L.ia[j]:L.ia[j+1]-1
         i = L.ja[k]
         if i==j
            d[perm[i]] = L.a[k]
         end
      end
   end
   return d
end

function getfactor(L::LDLtFactor)
   n = copy(L.n)
   ia = copy(L.ia)
   ja = copy(L.ja)
   a = copy(L.a)
   d = zeros(eltype(L.a),L.n)
   for j=1:L.n
      for k=ia[j]:ia[j+1]-1
         i = ja[k]
         if i==j
            d[i] = a[k]
            a[k] = 0.0
            break
         end
      end
   end
   T = SparseMatrixCSC(n,n,ia,ja,a)
   return d,T
end

function rank(L::LDLtFactor, tol=1e-9)
   return sum(abs.(diag(L)) .> tol)
end

"""
    L = nzstruct(A,perm)

Performs the symbolic factorization of `A`, i.e., analyzes the nonzero structure of
the Cholesky factor of `A` with a permutation `perm`. The indices of the factor are
stored in `L`. Storage for nonzero elements is also prepared in `L`.

### Examples
```jldoctest
julia> using AMD

julia> perm = amd(A)

julia> L = nzstruct(A,perm)
```
"""
function nzstruct(A::SparseMatrixCSC{Tv,Ti},perm::Vector{Ti}) where {Tv,Ti}
   L = LDLtFactor{Tv,Ti}(A.n,0,Ti[],Ti[],Ti[],Tv[],"")
   if(A.n < 1)
      throw(ArgumentError("Empty sparse matrix"))
   end
   if(A.n != length(perm))
      throw(DimensionMismatch("Size mismatch between the matrix and the permutation vector"))
   end
   if !has_all_diagonals(A.n, A.colptr, A.rowval)
      throw(ArgumentError("No all diagonals nor unsorted index"))
   end
   if !issymmetric(A)
      throw(ArgumentError("Unsymmetric sparse matrix"))
   end
   L.iperm = get_inverse_permutation(perm)
   ia = tril(A).colptr
   ja = tril(A).rowval
   pia = zeros(Ti,length(ia))
   pja = zeros(Ti,length(ja))
   find_permuted_matrix!(A.n,ia,ja,L.iperm,pia,pja)
   parent,L.nnz = Etgen(A.n,pia,pja)
   L.ia = zeros(Ti,A.n+1)
   L.ja = zeros(Ti,L.nnz)
   L.a = zeros(Tv,L.nnz)
   Symbolfac!(A.n,pia,pja,L.ia,L.ja)
   L.status = "SymbolicFactorization"
   return L
end

function nzstruct(As::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}},perm::Vector{Ti}) where {Tv,Ti}
   A = convert(SparseMatrixCSC,Symmetric(As))
   L = nzstruct(A,perm)
   return L
end

function ldlt!(L::LDLtFactor)
   if L.status != "SymbolicFactorization"
      throw(ArgumentError("Factor not prepared"))
   end
   LDLfac2!(L.n, L.ia, L.ja, L.a)
   L.status = "NumericalFactorization"
   return nothing
end

"""
    L = spchol(A)
    L = spchol(A, perm)

Performs sparse Cholesky factorization with Kachman's modification for rank deficiency.
It returns the `LDLtFactor` struct.
This function runs the sequence `perm=amd(A); L=nzstruct(A,perm); transfer!(A,L); ldlt!(L)`.
You can provide a permutation vector `perm` instead of using AMD.

If matrix `A` is not full-rank, this function returns a "generalized" factor with rows
and columns filled with zeros. Such null rows and columns are detected during
factorization as diagonal elements that are zero or nearly zero. Note that this process
might result in a numerically unstable factor and is not generally recommended. However,
there are particular situations where this kind of factor is useful, and users should
be aware of this behavior.

### Examples
```jldoctest
julia> L = spchol(A);
```
"""
function spchol(A,perm)
   L = nzstruct(A,perm)
   transfer!(A,L)
   ldlt!(L)
   return L
end

function spchol(A)
   perm = amd(A)
   return spchol(A,perm)
end

"""
    spsolve!(L, b)
    x = spsolve(L, b)
    x = L \\ b

Solve a system of linear equations using `L`, the Cholesky factor of `A`, and
the right-hand side vector or matrix `b`. Before calling these functions,
the sparse Cholesky factorization must have been completed with `spchol`.
The shorthand operator `\` is also available.

The function `spsolve!` updates `b` with the solution, while `spsolve` returns
the solution `x` without modifying `b`. The functions also accept `b` as
a matrix with multiple right-hand side vectors.

### Examples
```jldoctest
julia> L = spchol(A)      # has to be completed

julia> x = spsolve(L, b)  # or spsolve!(L, b)
```
"""
function spsolve!(L::LDLtFactor,b::Vector{Tv}) where {Tv}
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   if length(b) != L.n
      throw(DimensionMismatch("Size mismatch between the factor and RHS"))
   end
   x = zeros(Tv,L.n)
   x[L.iperm] = copy(b[1:L.n])
   LDLsolve!(L.n, L.ia, L.ja, L.a, x)
   b[1:L.n] = x[L.iperm]
   return nothing
end

function spsolve(L::LDLtFactor,b::Vector{Tv}) where {Tv}
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   if length(b) != L.n
      throw(DimensionMismatch("Size mismatch between the factor and RHS"))
   end
   x = zeros(Float64,L.n)
   x[L.iperm] = copy(b[1:L.n])
   LDLsolve!(L.n, L.ia, L.ja, L.a, x)
   x[1:L.n] = x[L.iperm]
   return x
end

# inefficient implementation for multiple RHS: future work
function spsolve!(L::LDLtFactor,B::Matrix{Tv}) where {Tv}
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   N = size(B)[1]
   M = size(B)[2]
   if N != L.n
      throw(DimensionMismatch("Size mismatch between the factor and RHS"))
   end
   x = zeros(Tv,L.n)
   for i=1:M
      x[L.iperm] = Vector{Tv}(copy(B[1:L.n,i]))
      LDLsolve!(L.n, L.ia, L.ja, L.a, x)
      B[1:L.n,i] = x[L.iperm]
   end
   return nothing
end

function spsolve(L::LDLtFactor,B::Matrix{Tv}) where {Tv}
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   N = size(B)[1]
   M = size(B)[2]
   if N != L.n
      throw(DimensionMismatch("Size mismatch between the factor and RHS"))
   end
   X  =zeros(Tv,N,M)
   x = zeros(Tv,N)
   for i=1:M
      x[L.iperm] = Vector{Tv}(copy(B[1:L.n,i]))
      LDLsolve!(L.n, L.ia, L.ja, L.a, x)
      X[1:L.n,i] = x[L.iperm]
   end
   return X
end

function (\)(L::LDLtFactor, b::Vector{Tv}) where {Tv}
   return spsolve(L,b)
end

function (\)(L::LDLtFactor, B::Matrix{Tv}) where {Tv}
   return spsolve(L,B)
end

"""
    d = spdet(L)
    d = splogdet(L)

Calculates the determinant of a positive semi-definite matrix `A` using
its Cholesky factor `L`. The factor `L` must have been computed with `spchol`.
The log-determinant of A is also available through `splogdet`.

If `A` is not full-rank and the factor has zero-filled diagonal elements,
this function skips those diagonal elements and calculates the determinant
using only the nonzero diagonal elements.

### Examples
```jldoctest
julia> L = spchol(A)  # has to be completed

julia> d = spdet(L)   # determinant of A

julia> logd = splogdet(L)  # log-determinant of A
```
"""
function spdet(L::LDLtFactor)
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   return get_determinant(L.n,L.ia,L.ja,L.a)
end

function splogdet(L::LDLtFactor)
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   return get_log_determinant(L.n,L.ia,L.ja,L.a)
end

"""
    spinv!(L)
    spinv!(L, A)

Calculates the "sparse inverse" (also known as the "selected inverse") of `A`.
The function `spinv!` calculates the inverse elements of `A` corresponding to
the nonzero elements in `L`, and updates `L` with the inverse elements.
If the original matrix `A` is provided, the function updates both `L` and `A`
with the inverse elements in their nonzero elements.

Once this function is called and `L` is updated, users can no longer use that `L`
for factor-related functions such as `spsolve`, `spdet`, and so on.

### Examples
```jldoctest
julia> L = spchol(A)   # has to be completed

julia> spinv!(L, A)    # update L and A with inverse elements
```
"""
function spinv!(L::LDLtFactor)
   if L.status == "SparseInversion"
      throw(ArgumentError("Already inverted"))
   end
   if L.status != "NumericalFactorization"
      throw(ArgumentError("No factor in the object"))
   end
   Sparseinv!(L.n, L.ia, L.ja, L.a)
   L.status="SparseInversion"
   return nothing
end

function spinv!(L::LDLtFactor, A::SparseMatrixCSC)
   if L.n != A.n || A.n != A.m
      throw(DimensionMismatch("Size mismatch between two matrices"))
   end
   spinv!(L)
   save_factor_to_sparse!(L.n, A.colptr, A.rowval, A.nzval, L.iperm, L.ia, L.ja, L.a)
   return nothing
end

function spinv!(L::LDLtFactor, As::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
   if L.n != As.data.n || As.data.n != As.data.m
      throw(DimensionMismatch("Size mismatch between two matrices"))
   end
   spinv!(L)
   save_factor_to_sparse!(L.n, As, L.iperm, L.ia, L.ja, L.a)
   return nothing
end
