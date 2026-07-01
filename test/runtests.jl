using LinearAlgebra
using SparseArrays
using AMD
using OldFashionedSparseSolver
using Test

@testset "primitive tests" begin
   n = 10
   x = ones(n,1)
   A = sparse(Symmetric( -1.0*(triu(rand(n,n),1).>0.6) + 1.0*n*Matrix(I,n,n) ))
   b = A*x
   perm = amd(A)
   L = nzstruct(A,perm)
   transfer!(A,L)
   ldlt!(L)
   sol = L \ b
   @test A*sol ≈ b
   #spinv!(L, A)
end

@testset "batch function tests" begin
   n = 10
   x = ones(n,1)
   A = sparse(Symmetric( -1.0*(triu(rand(n,n),1).>0.6) + 1.0*n*Matrix(I,n,n) ))
   b = A*x
   L = spchol(A)
   sol = L \ b
   @test A*sol ≈ b

   perm = amd(A)
   L1 = spchol(A, perm)
   sol1 = L1 \ b
   @test sol1 ≈ sol
   #spinv!(L, A)
end

@testset "inverse tests" begin
   A = [
      4.0 1.0 0.0 0.0
      1.0 5.0 2.0 0.0
      0.0 2.0 6.0 1.0
      0.0 0.0 1.0 3.0
   ]
   sA = sparse(A)
   original_sA = copy(sA)
   invA = inv(A)

   L = spchol(sA)
   @test spinv!(L, sA) === nothing

   I, J, _ = findnz(original_sA)
   @test all(sA[i,j] ≈ invA[i,j] for (i,j) in zip(I,J))
end

@testset "various tests" begin
   dimlist = [10,50,100]
   threshold = 0.6
   niter = 20

   # positive definite cases
   for n in dimlist;
      for iter in 1:niter
         x = ones(n,1)
         A = sparse(Symmetric( -1.0*(triu(rand(n,n),1).>threshold) + 1.0*n*Matrix(I,n,n) ))
         C = Matrix(A)
         b = A*x
         L = spchol(A)
         solx = L \ b
         soly = C \ b
         @assert isapprox(solx,soly)
         @test A*solx ≈ b

         spinv!(L, A)
         S=(C.!=0.0)*1.0
         q1=diag(Matrix(A))
         q2=diag(inv(C).*S)
         @assert norm(q1-q2)<1e-1
      end
   end
end

@testset "determinant tests" begin
   A = sparse([
      4.0 1.0 0.0
      1.0 3.0 1.0
      0.0 1.0 2.0
   ])

   L = spchol(A)

   detA, rankA = spdet(L)
   logdetA, logrankA = splogdet(L)

   @test rankA == size(A, 1)
   @test logrankA == size(A, 1)
   @test detA ≈ det(Matrix(A))
   @test logdetA ≈ logdet(Symmetric(Matrix(A)))
   @test logdetA ≈ log(detA)

   perm = amd(A)
   Lsym = nzstruct(A, perm)

   @test_throws ArgumentError spdet(Lsym)
   @test_throws ArgumentError splogdet(Lsym)
end

@testset "rank deficient determinant tests" begin
   A = sparse([
      1.0 1.0 0.0
      1.0 1.0 0.0
      0.0 0.0 2.0
   ])

   L = spchol(A)
   d = diag(L)
   nzdiag = d[d .!= 0.0]

   detA, rankA = spdet(L)
   logdetA, logrankA = splogdet(L)

   @test rankA == length(nzdiag)
   @test logrankA == length(nzdiag)
   @test detA ≈ prod(nzdiag)
   @test logdetA ≈ sum(log.(nzdiag))
end

@testset "sparse inverse tests" begin
   # simple matrix
   A = sparse([
      4.0 1.0 0.0
      1.0 3.0 1.0
      0.0 1.0 2.0
   ])

   A0 = copy(A)
   L = spchol(A)

   @test spinv!(L, A) === nothing

   # numerical and logical checks
   mask = Matrix(A0 .!= 0.0)
   expected = inv(Matrix(A0)) .* mask

   @test Matrix(A) ≈ expected
   @test L.status == "SparseInversion"
end

@testset "sparse inverse dimension checks" begin
   A = sparse([
      4.0 1.0 0.0
      1.0 3.0 1.0
      0.0 1.0 2.0
   ])

   L = spchol(A)
   B = sparse(Matrix(I, 2, 2))

   @test_throws DimensionMismatch spinv!(L, B)
end

@testset "sparse inverse corner-case checks" begin
   A = sparse([
       1.2   0.0  -0.1  -0.1  0.0  0.0  0.0
       0.0   1.1  -0.1   0.0  0.0  0.0  0.0
      -0.1  -0.1   1.2   0.0  0.0  0.0  0.0
      -0.1   0.0   0.0   1.1  0.0  0.0  0.0
       0.0   0.0   0.0   0.0  1.0  0.0  0.0
       0.0   0.0   0.0   0.0  0.0  2.0  1.0
       0.0   0.0   0.0   0.0  0.0  1.0  2.0 
   ])
   n = size(A, 1)
   B = copy(A)
   perm = collect(1:n)
   L = spchol(A, perm)
   spinv!(L)
   transfer!(L, B)

   C = inv(Matrix(A))

   for j=1:n
      for i=j:n
         if !(B[i,j] ≈ 0.0)
            @test B[i,j] ≈ C[i,j]
         end
      end
   end
end
