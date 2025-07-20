
% Generating problems of given TLS classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function generates the model and observation matrices A and B such
% that the problem AX ≈ B is of the specified TLS class. It also outputs
% the right singular matrix V corresponding to the extended matrix [B,A].


%   INPUT parameters
% * m ... number of rows of matrices A and B
% * n ... number of columns of matrix A
% * d ... number of columns of matrix B
% * class ... desired TLS class of the resulting problem
%         = "S", "F1", "F2" or "F3"
% * S ... column vector of singular values of the extended matrix [B,A]
%     ! Note that it may need to satisfy restrictions on multiplicity of !
%     ! some singular values posed by the desired TLS class.             !
% * tol ... tolerance parameter used to determine the mutiplicity of singular values

%   OUTPUT parameters
% * A ... (m x n) real matrix; the model matrix
% * B ... (m x d) real matrix; the observation matrix
% * V ... ((n+d) x (n+d)) real orthogonal matrix, right singular space of
%     the extended matrix [B,A] satisfying conditions of the desired class

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [A,B,V] = TLSclass(m,n,d,class,S,tol)

% SVD of the extended matrix [B,A]
U = randn(m); [U,~] = qr(U);
V = randn(n+d); [V,~] = qr(V);
D = zeros(m,n+d); D(1:size(S,1),1:size(S,1)) = diag(S);

% Multiplicities of sigma_n+1
% left multiplicity
l = 0;
i = n; 
while ((S(i)-S(n+1))/S(n+1)) < tol 
    l = l + 1;
    i = i-1;
    if i == 0
        break
    end
end

% right multiplicity
r = 1;             % including current singular value                      
i = n+2;
while ((S(i)-S(n+1))/S(n+1)) < tol 
    r = r + 1;
    i = i+1;
    if i == n+d+1
        break
    end
end

% Eliminating columns of V12 to reach given class
if class == "F1"                    % rank(V12) = r , rank(V13) = d-r (full)
    if l ~= 0
        for i = 0:(l-1)             % zero out first l columns of V12 => rank=r
                v = V(1:(d+l-i),n-i);
                e = zeros(d+l-i,1); e(end) = 1;
                q = v + sign(v(end)) * norm(v) * e;
                q = q / norm(q);
                h = eye(d+l-i) - 2*(q*q');
                H = eye(n+d);
                H(1:(d+l-i),1:(d+l-i)) = h;
                V = H * V;
                % zero out entries that should have been eliminated by the
                % Householder reflection but, due to FPA, are not exactly zero
                % (instead, they are below the machine epsilon threshold)
                V(1:(d+l-i-1),n-i) = zeros(d+l-i-1,1);
        end
    end

elseif class == "F3"                % rank(V12) > r , rank(V13) < d-r
    if d > 1
        v = V(1:(d+1),end);         % zero out last column of V13 => rank=d-r-1
        e = zeros(d+1,1); e(end) = 1;
        q = v + sign(v(end)) * norm(v) * e;
        q = q / norm(q);
        h = eye(d+1) - 2*(q*q');
        H = eye(n+d);
        H(1:(d+1),1:(d+1)) = h;
        V = H * V;
        % zero out entries that should have been eliminated by the
        % Householder reflection but, due to FPA, are not exactly zero
        % (instead, they are below the machine epsilon threshold)
        V(1:d,end) = zeros(d,1);
    end

elseif class == "S"                 % rank([V12,V13]) < d
    for i = 0:l                     % zero out last l+1 columns
        v = V(1:(d+l-i+1),end-i);
        e = zeros(d+l-i+1,1); e(end) = 1;
        q = v + sign(v(end)) * norm(v) * e;
        q = q / norm(q);
        h = eye(d+l-i+1) - 2*(q*q');
        H = eye(n+d);
        H(1:(d+l-i+1),1:(d+l-i+1)) = h;
        V = H * V;
        % zero out entries that should have been eliminated by the
        % Householder reflection but, due to FPA, are not exactly zero
        % (instead, they are below the machine epsilon threshold)
        V(1:(d+l-i),end-i) = zeros(d+l-i,1);
    end
end

% Constructing the data matrices from the computed SVD of the extended matrix
BA = U * D * V';
A = BA(:,(d+1):end);
B = BA(:,1:d);