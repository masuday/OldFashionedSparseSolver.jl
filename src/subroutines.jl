
function get_inverse_permutation(perm)
   n = length(perm)
   iperm = zero(perm)
   for i=1:n
      iperm[perm[i]] = i
   end
   return iperm
end

function has_all_diagonals(neq, ia, ja)
   for i=1:neq
      okay = false
      kfst = ia[i]
      klst = ia[i+1]-1
      for k=kfst:klst
         j = ja[k]
         if i == j
            okay = true
         end
      end
      if !okay
         return false
      end
   end
   return true
end

function get_size_for_graph(neq, ia, ja)
   gsz = 0
   for i=1:neq
      kfst = ia[i]
      klst = ia[i+1]-1
      # number of nonzeros within a column
      ne = klst - kfst + 1
      # count
      for k=kfst:klst
         j = ja[k]
         if i != j
            # upper and lower elements
            gsz = gsz + 2
         end
      end
   end
   return gsz
end

function find_permuted_graph!(neq, ia, ja, iperm, pxadj, padjncy)
   # initialization
   cnt = zeros(Int64,neq)

   # count up the elements
   # i, j  : original indices
   # ii,jj : permuted indices
   for i=1:neq
      ii = iperm[i]
      for k=ia[i]:ia[i+1]-1
         j = ja[k]
         jj = iperm[j]
         mi = min(ii,jj)
         mj = max(ii,jj)
         if ii != jj
            cnt[mi] = cnt[mi]+1
            cnt[mj] = cnt[mj]+1
         end
      end
   end
   # set pia and clear cnt(:)
   pxadj[1] = 1
   for i=1:neq
      pxadj[i+1] = pxadj[i] + cnt[i]
      cnt[i] = 0
   end
   # set pja
   # i, j  : original indices
   # ii,jj : permuted indices
   for i=1:neq
      ii = iperm[i]
      for k=ia[i]:ia[i+1]-1
         j = ja[k]
         jj = iperm[j]
         mi = min(ii,jj)
         mj = max(ii,jj)
         if ii != jj
            padjncy[pxadj[mi]+cnt[mi]] = mj
            padjncy[pxadj[mj]+cnt[mj]] = mi
            cnt[mi] = cnt[mi]+1
            cnt[mj] = cnt[mj]+1
         end
      end
   end
   # sort index
   sort_ja!(pxadj,padjncy)
   return nothing
end

function find_permuted_matrix!(neq, ia, ja, iperm, pia, pja)
   # initialization
   cnt = zeros(Int64,neq)

   # count up the elements
   # i, j  : original indices
   # ii,jj : permuted indices
   for i=1:neq
      ii = iperm[i]
      for k=ia[i]:ia[i+1]-1
         j = ja[k]
         jj = iperm[j]
         mi = min(ii,jj)
         cnt[mi] = cnt[mi]+1
      end
   end
   # set pia and clear cnt(:)
   pia[1] = 1
   for i=1:neq
      pia[i+1] = pia[i] + cnt[i]
      cnt[i] = 0
   end
   # set pja
   # i, j  : original indices
   # ii,jj : permuted indices
   for i=1:neq
      ii = iperm[i]
      for k=ia[i]:ia[i+1]-1
         j = ja[k]
         jj = iperm[j]
         mi = min(ii,jj)
         mj = max(ii,jj)
         pja[pia[mi]+cnt[mi]] = mj
         cnt[mi] = cnt[mi]+1
      end
   end
   # sort index
   sort_ja!(pia,pja)
   return nothing
end

function sort_ja!(ia,ja)
   neq = length(ia)-1
   for i=1:neq
      jfst=ia[i]
      jlst=ia[i+1]-1
      if jlst>jfst
         ja[jfst:jlst]=sort(ja[jfst:jlst])
      end
   end
   return nothing
end

function load_sparse_to_factor!(neq,Aia,Aja,Aa,iperm,Lia,Lja,La)
   posij = 0
   for jj=1:neq
      j = iperm[jj]
      kfst = Aia[jj]
      klst = Aia[jj+1]-1
      for kk=kfst:klst
         ii = Aja[kk]
         if ii>=jj
            i = iperm[ii]
            val = Aa[kk]
            posij = set_value!(Lia,Lja,La,val,i,j)
         end
      end
   end
   return nothing
end

function save_factor_to_sparse!(neq,Aia,Aja,Aa,iperm,Lia,Lja,La)
   posij = 0
   for jj=1:neq
      j = iperm[jj]
      kfst = Aia[jj]
      klst = Aia[jj+1]-1
      for kk=kfst:klst
         ii = Aja[kk]
         i = iperm[ii]
         posij,val = get_value(Lia,Lja,La,i,j)
         Aa[kk] = val
      end
   end
   return nothing
end

function save_factor_to_sparse!(neq,As::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}},iperm,Lia,Lja,La) where {Tv,Ti}
   posij = 0
   for jj=1:neq
      j = iperm[jj]
      kfst = As.data.colptr[jj]
      klst = As.data.colptr[jj+1]-1
      for kk=kfst:klst
         ii = As.data.rowval[kk]
         i = iperm[ii]
         posij,val = get_value(Lia,Lja,La,i,j)
         As.data.nzval[kk] = val
      end
   end
   return nothing
end

function set_value!(ia,ja,a,val,i,j)
   ii = min(i,j)
   jj = max(i,j)
   kfst = ia[ii]
   klst = ia[ii+1]-1
   for k=kfst:klst
      if jj==ja[k]
         posij = k
         a[posij] = val
         return posij
      end
   end
   posij = 0
   return posij
end

function get_value(ia,ja,a,i,j)
   ii = min(i,j)
   jj = max(i,j)
   kfst = ia[ii]
   klst = ia[ii+1]-1
   for k=kfst:klst
      if jj==ja[k]
         posij = k
         val = a[posij]
         return posij,val
      end
   end
   posij = zero(eltype(ja))
   val = zero(eltype(a))
   return posij,val
end

function show_nonzero_structure(neq, ia, ja, a, showfull::Bool=false)
   for i=1:neq
      for j=1:neq
         if showfull
            posij,val=get_value(ia,ja,a,i,j)
         else
            if i>=j
               posij,val=get_value(ia,ja,a,i,j)
            else
               posij = 0
            end
         end
         if posij>0
            print("o ")
         else
            print(". ")
         end
      end
      println("")
   end
end

function RowTrav!(ia, ja, i, j, posij, nextj, link, pos)
   if j==0
      i = i + 1
      j = i
      posij = ia[i]
   else
      j = nextj
      if j==0
         return i,j,posij,nextj
      end
      posij = pos[j]
   end
   nextj = link[j]
   link[j] = zero(eltype(link))
   nextdown = posij + 1
   if(nextdown < ia[j+1])
      pos[j] = nextdown
      id = ja[nextdown]
      link[j] = link[id]
      link[id] = j
   end
   return i,j,posij,nextj
end

function Etgen(neq, ia, ja)
   # initialization
   i = 0
   j = 0
   link = zeros(eltype(i),neq)
   pos = zeros(eltype(i),neq)
   posij = 0
   nextj = 0
   touched = zeros(eltype(i),neq)

   # returning values
   parent = zeros(eltype(i),neq)
   nLNZ = 0

   # row-traversal
   for ix=1:neq
      while(true)
         # traverse
         i,j,posij,nextj = RowTrav!(ia,ja,i,j,posij,nextj,link,pos)
         jj = j
         if jj==0
            break
         end
         if i==j
            #! diagonals (i,i)
            nLNZ = nLNZ + 1
            touched[i] = i
         else
            #! off diagonals (i,j)
            js = j
            while(touched[js]!=i)
               touched[js] = i
               nLNZ = nLNZ + 1
               if parent[js]==0
                  parent[js] = i
                  break
               end
               js = parent[js]
            end
         end
      end
   end
   return parent,nLNZ
end

function Merge!(Bia, Bja, j, k, ma)
   # start
   m = k
   # Loop over elements in column j of B
   fst = Bia[j]
   lst = Bia[j+1]-1
   for ii=fst:lst
      i = Bja[ii]
      # NOTE: the following statement is needed to finish successfully.
      if i < k
         continue
      end
      # search for m and m1 with m<i<=m1
      m1 = m
      while(i > m1)
         m = m1
         m1 = ma[m]
      end
      if i != m1
         # insert i in ma
         ma[m] = i
         ma[i] = m1
      end
      m = i
   end
   return nothing
end

function Makecol!(k, ma, neq, Lia, Lja)
   # init
   if k==1
      Lia[1] = 1
   end
   ii = Lia[k]
   m = k
   # gather row indices from a linked-list (ma) into Lja
   #  - position ii increment
   #  - ma(m) initialized with a "huge" value (neq+1)
   while(m <= neq)
      Lja[ii] = m
      ii = ii + 1
      mt = ma[m]
      ma[m] = neq+1
      m = mt
   end
   Lia[k+1] = ii
   return nothing
end

function Symbolfac!(neq, Aia, Aja, Lia, Lja)
   # initialization
   bs = zeros(eltype(Lja),neq)
   ma = ones(eltype(Lja),neq)*eltype(Lja)(neq+1)
   jt = 0

   # main loop on column (k) of A
   for k=1:neq
      # compute the structure of the k-th column
      Merge!(Aia,Aja, k,k,ma)
      j = bs[k]
      while(j != 0)
         Merge!(Lia,Lja, j,k,ma)
         jt = bs[j]
         bs[j] = 0
         j = jt
      end
      # set up the k-th column of L
      Makecol!(k, ma, neq, Lia, Lja)
      # update the baby sitter
      if k != neq
         # j is the parent of k
         j = Lja[ Lia[k]+1 ]
         while(j != 0)
            jt = j
            j = bs[j]
         end
         if jt>0
            bs[jt] = k
         end
      end
   end
   return nothing
end

function LDLfac2!(neq,Lia,Lja,La; tol::Float64=1e-9, dmin::Float64=1e-8)
   # initialization
   k = 0
   j = 0
   nextj = 0
   poskj = 0
   link = zeros(eltype(Lja),neq)
   pos = zeros(eltype(Lja),neq)
   kids = zeros(eltype(Lja),2,neq)
   accum = zeros(eltype(La),neq)
   irank = 0

   # process column kx
   for kx=1:neq
      # initialize accum
      ifst = Lia[kx]
      ilst = Lia[kx+1]-1
      for ii=ifst:ilst
         ipos = Lja[ii]
         accum[ipos] = La[ii]
      end
      # pre-traversal
      j = 0
      nkids = 0
      while(true)
         #jj = RowTrav(Lia,Lja,k,j,poskj,link,pos,nextj)
         k,j,poskj,nextj = RowTrav!(Lia,Lja,k,j,poskj,nextj,link,pos)
         jj = j
         if jj==0
            break
         end
         if j != k
            nkids = nkids + 1
            kids[1,nkids] = j
            kids[2,nkids] = poskj
         end
      end
      # traversal
      for kid=1:nkids
         # load values
         j = kids[1,kid]
         poskj = kids[2,kid]
         # subtract L(k:n,j) from L(k:n,k)
         Dj = La[Lia[j]]
         Lkj = La[poskj]
         ifst = poskj
         ilst = Lia[j+1]-1
         for ii=ifst:ilst
            i = Lja[ii]
            accum[i] = accum[i] - Dj*Lkj*La[ii]
         end
      end
      # move L(k:n,k) from accum to L, adjusting its components
      ifst = Lia[kx]
      ilst = Lia[kx+1]-1
      # diagonal elements (Dk) and its inverse (Lkkinv)
      ii = ifst
      i = Lja[ii]
      Akk = La[ii]
      Dk = accum[i]
      accum[i] = 0.0
      # Dk and Lkkinv modified with the Kachman's method
      # NOTE: irank will be incremented if Dk is not zero.
      Dk,Lkkinv,arank = Kachman_modification(Akk,Dk, tol,dmin)
      irank = irank + arank
      La[ii] = Dk
      # off-diagonals
      for ii=ifst+1:ilst
         i = Lja[ii]
         La[ii] = Lkkinv*accum[i]
         accum[i] = 0.0
      end
   end
   return nothing
end

function Kachman_modification(Akk,Dk,tol,dmin)
   if Akk > dmin
      if Dk > tol*Akk
         # accept Dk
         arank = 1
         Dkinv = 1.0/Dk
      else
         arank = 0
         Dk = 0.0
         Dkinv = 0.0
      end
   else
      arank = 0
      Dk = 0.0
      Dkinv = 0.0
   end
   return Dk,Dkinv,arank
end

function LDLsolve!(neq,Lia,Lja,La, b)
   for j=1:neq
      bj = b[j]
      ifst = Lia[j]+1
      ilst = Lia[j+1]-1
      for ii=ifst:ilst
         i = Lja[ii]
         b[i] = b[i] - bj*La[ii]
      end
   end
   for j=1:neq
      Ldiag = La[Lia[j]]
      if Ldiag != 0.0
         b[j] = b[j]/Ldiag
      end
   end
   for j=neq:-1:1
      bj = b[j]
      ifst = Lia[j]+1
      ilst = Lia[j+1]-1
      for ii=ifst:ilst
         i = Lja[ii]
         bj = bj - b[i]*La[ii]
      end
      b[j] = bj
   end
   return nothing
end

function get_determinant(neq,Lia,Lja,La)
   rank = 0
   det = 1.0
   for i=1:neq
      # diagonal elements
      k = Lia[i]
      #j = Lja[k]
      diag = La[k]
      if diag != 0.0
         det = det * diag
         rank = rank + 1
      end
   end
   return det,rank
end

function get_log_determinant(neq,Lia,Lja,La)
   rank = 0
   logdet = 0.0
   for i=1:neq
      # diagonal elements
      k = Lia[i]
      #j = Lja[k]
      diag = La[k]
      if diag != 0.0
         logdet = logdet + log(diag)
         rank = rank + 1
      end
   end
   return logdet,rank
end

function Sparseinv!(neq,Lia,Lja,La)
   # initialization
   TMPLj = zeros(eltype(La),neq)
   TMPZj = zeros(eltype(La),neq)
   touched = zeros(eltype(Lja),neq)
   istouched = falses(neq)
   ntouched = 0

   # column-wise update
   for j=neq:-1:1
      ntouched = 0
      kfst = Lia[j]
      klst = Lia[j+1]-1
      Dj = La[kfst]
      if Dj<=0.0
         continue
      end

      ntouched = ntouched + 1
      touched[ntouched] = j
      istouched[j] = true
      TMPZj[j] = 1.0/Dj
      for k=kfst+1:klst
         i = Lja[k]
         TMPLj[i] = La[k]
      end

      for kk=kfst+1:klst
         k = Lja[kk]
         ifst = Lia[k]
         ilst = Lia[k+1]-1
         # If TMPZj[i] was written during this column,
         # then i must be recorded and cleared before
         # the next column.
         # diagonal element of column k (k==i)
         i = Lja[ifst]
         if !istouched[i]
            ntouched = ntouched + 1
            touched[ntouched] = i
            istouched[i] = true
         end
         TMPZj[i] = TMPZj[i] - TMPLj[i]*La[ifst]
         # off diagonal elements
         for ii=ifst+1:ilst
            i = Lja[ii]
            if !istouched[i]
               ntouched = ntouched + 1
               touched[ntouched] = i
               istouched[i] = true
            end
            TMPZj[i] = TMPZj[i] - TMPLj[k]*La[ii]
            if !istouched[k]
               ntouched = ntouched + 1
               touched[ntouched] = k
               istouched[k] = true
            end
            TMPZj[k] = TMPZj[k] - TMPLj[i]*La[ii]
         end
      end
      # diagonal update
      for kk=kfst+1:klst
         k = Lja[kk]
         TMPZj[j] = TMPZj[j] - TMPLj[k]*TMPZj[k]
         # reset
         TMPLj[k] = 0.0
      end
      # save
      for kk=kfst:klst
         k = Lja[kk]
         La[kk] = TMPZj[k]
      end
      # reset all touched temporary values
      # NOTE: TMPZj is not only written at entries
      # that will be saved for the current column.
      # So clearing only saved entries is insufficient.
      for kk=1:ntouched
         k = touched[kk]
         TMPZj[k] = 0.0
         istouched[k] = false
      end
   end
   return nothing
end
