
% Construction of the TLS solution using the GKB approximation of the core problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   This function approximates the core problem corresponding to a linear
% system AX ≈ B using the Golub-Kahan iterative bidiagonalization (our
% implementation "blockIGKB.m"), then computes the TLS solution of the core
% problem using classical TLS agorithm (our implemetation "blockTLS.m")
% and finally projects the core problem solution X_core to approximate the
% TLS solution X of the original problem AX ≈ B. 
%
%
%   INPUT parameters
% * A ... (m × n) real matrix; the model matrix
% * B ... (m x d) real matrix; the data matrix
% * tol ... tolerance parameter used to identify multiple singular values in TLS
% * tol0 ... tolerance parameter used to identify zero singular values of B in GKB
% * tol1 ... tolerance parameter used to detect deflation in GKB
% * iter ... maximum number of iterations in GKB
% * reorthog ... integer specifying required reorthogonalization method in GKB
%            = "0" ... without reorthogonalization
%            = "1" ... one-sided reorthogonalization
%            = "2" ... full reorthogonalization
% 
%   OUTPUT parameters
% * X ... (n x d) real matrix; TLS or non-generic solution to AX ≈ B
%         computed by backward transformation from a TLS solution X_core of
%         the core problem
% * orig_class ... determined class of the computed core problem solution X_core
%                  according to TLS theory
%              = "S" ... the solution is of the class S
%              = "F1" ... the solution is of the class F1
%              = "F2" ... the solution is of the class F2
%              = "F3" ... the solution is of the class F3
% * flag ... integer indicating whether the computed core problem solution
%            is a TLS or non-generic solution
%        = "0" ... X_core is a classical TLS solution to the core problem
%        = "i", i ≥ 1 ... X_core is a non-generic solution & singular vectors
%                         corresponding to i distinct singular values must
%                         be added to the subspace to construct the solution

%   Optional OUTPUT parameters corresponding to X_core
% * classes ... row vector of all classes (in each run of the while loop)
% * Vmin ... resulting transformed right singular subspace
% * S ... singular values of [B1,L] computed by "svd" function
% * V ... right singular space computed by "svd" function
% * left_multiplicities ... row vector of left multiplicities of all used
%                       singular values
% * right_multiplicities ... row vector of right multiplicities of all used
%                        singular values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X,orig_class,flag, classes,Vmin,S,V,lelf_multiplicities,right_multiplicities] = core (A,B,tol,tol0,tol1,iter,reorthog)

% Core problem construction
[Q,~,L,B1,Vb, ~,~,~,~] = blockIGKB(A,B,tol0,tol1,iter,reorthog);

% TLS solution of the core problem
[X_core,orig_class,flag, classes,Vmin,S,V,left_multiplicities,right_multiplicities] = blockTLS(L,B1,tol);

% Backward transformation
X = zeros(size(Q,2),size(Vb,2)); X(1:size(X_core,1),1:size(X_core,2)) = X_core;
X = Q * X * Vb';

end