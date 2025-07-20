
% Block Golub-Kahan iterative bidiagonalization and core reduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   This function implements a block version of the Golub-Kahan iterative
% bidiagonalization algorithm. Given matrices A and B, along with the maximum
% number of iterations, a desired tolerance to either identify zero singular
% values or detect deflation in block size, and an indicator specifying
% required reorthogonalization method, the function computes orthonormal
% matrices Q and P, which represent the bases of the Krylov spaces K_k(A'A,A'B)
% and K_k(AA',B), respectivelly, and a block bidiagonal matrix L.
%   Additionally, the algorithm also outputs matrices B1 and Vb, which are
% needed to construct the core problem and a TLS solution to AX ≈ B later on.
% The corresponding core problem is then of the form LX_core ≈ B1.
% The function also provides vectors that track the loss of orthogonality
% in the computed orthogonal matrices and monitor the deflation in size of
% the lower or upper diagonal blocks in L. 


%   INPUT parameters
% * A ... (m × n) real matrix; the model matrix
% * B ... (m x d) real matrix; the data matrix
% * tol0 ... tolerance parameter used to identify zero singular values of B
% * tol1 ... tolerance parameter used to detect upper and lower deflation
% * iter ... maximum number of iterations (serves as a stopping criterion
%            in case the algorithm doesn't detect deflation)
% * reorthog ... integer specifying required reorthogonalization method
%            = "0" ... without reorthogonalization
%            = "1" ... one-sided reorthogonalization
%            = "2" ... full reorthogonalization

%   OUTPUT parameters
% * Q ... orthogonal matrix, bases of K_iter(A'A,A'B)
% * P ... orthogonal matrix, bases of K_iter(AA',B)
% * L ... block bidiagonal matrix
% * B1 ... RHS of the core problem
% * Vb ... (d x d) unitary matrix, right singular space of B used later for
%          transformation of the core problem solution to TLS solution

%   Optional OUTPUT parameters
% * lossOG_P ... row vector, loss of orthogonality in P in each
%                iteration computed as ||I - P'P||
% * lossOG_Q ... row vector, loss of orthogonality in Q in each
%                iteration computed as ||I - Q'Q||
% * lower_deflation ... row vector, entries denoting size of lower
%                       diagonal block (R) in L in each iteration
% * upper_deflation ... row vector, entries denoting size of upper
%                       diagonal block (D) in L in each iteration

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Q,P,L,B1,Vb, lossOG_P,lossOG_Q,lower_deflation,upper_deflation] = blockIGKB(A,B,tol0,tol1,iter,reorthog)

% Preprocessing of the RHS: eliminating excess columns from B using SVD
[Ub,Sb,Vb] = svd(B);
for k = 1:min(size(Sb,1),size(Sb,2))
    if Sb(k,k) < tol0
        Sb = Sb(:,1:k-1);
        break
    end
end
B1 = Ub*Sb;

% Inicialization
deflation_control = min(size(Sb,2),size(A,2));
[p,R] = qr(B1,"econ");
B1 = R;
q = zeros(size(A,2),size(Sb,2));
Q = [];
P = p;
L = [];
lossOG_Q = [];
lossOG_P = norm(eye(size(P,2))-P.'*P);
lower_deflation = rank(R);
upper_deflation = [];

% Main iteration
for k = 1:iter
    if deflation_control > 0

        % Computing current Q and D
        q = A.' * p - q * R.';
        % Performing full reorthogonalization if requested
        if reorthog == 2 && k ~= 1
            R = Q.'*q;
            q = q - Q*R;
        end
        [q,D] = qr(q,"econ");
        % Determining upper deflation
        deflation = [];
        for i = 1:size(D,1)
            if max(abs(D(i,:))) < tol1
                deflation = [deflation,i];
            end
        end
        upper_deflation = [upper_deflation, size(D,1)-size(deflation,2)];
        D(deflation,:) = [];
        q(:,deflation) = [];
        deflation_control = deflation_control - size(deflation,2);
        % Updating matrices L and Q
        Q = [Q,q];
        lossOG_Q = [lossOG_Q, norm(eye(size(Q,2)) - Q.'*Q)];
        update = [zeros(size(L,1)-size(D,2),size(D,1)); D.'];
        L = [L , update];


        % Computing current P and R
        p = A * q - p * D.';
        % Performing full or one-sided reorthogonalization if requested
        if reorthog > 0
            D = P.'*p;
            p = p - P*D;
        end
        [p,R] = qr(p,"econ");
        % Determining lower deflation
        deflation = [];
        for i = 1:size(R,1)
            if max(abs(R(i,:))) < tol1
                deflation = [deflation,i];
            end
        end
        lower_deflation = [lower_deflation, size(R,1) - size(deflation,2)];
        R(deflation,:) = [];
        p(:,deflation) = [];
        deflation_control = deflation_control - size(deflation,2);
        % Updating matrices L and P
        P = [P, p];
        lossOG_P = [lossOG_P, norm(eye(size(P,2)) - P.'*P)];
        update = [zeros(size(R,1),size(L,2)-size(R,2)), R];
        L = [L ; update];
    end
end

% Expanding B1 by an apropriate number of zeros to correspond to the size of L
B1 = [B1;zeros(size(L,1)-size(B1,1),size(B1,2))];

end