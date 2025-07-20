
% Block TLS algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   This function implements a block version of the classical Total Least
% Squares (TLS) algorithm. Given a model matrix A and an observation matrix B,
% along with a desired tolerance to identify the multiplicity of singular
% values, the function computes a TLS solution matrix X to the approximation
% problem AX ≈ B, if it exists. Otherwise it computes a so called
% non-generic solution. Additionally, the algorithm determines the class of
% the computed solution. 

 
%   INPUT parameters
% * A ... (m × n) real matrix; the model matrix
% * B ... (m x d) real matrix; the data matrix
% * tol ... tolerance parameter used to determine the mutiplicity of singular values
 
%   OUTPUT parameters
% * X ... (n x d) real matrix; computed TLS or non-generic solution to AX ≈ B
% * orig_class ... determined class of the computed solution X according to TLS theory
%              = "S" ... the solution is of the class S
%              = "F1" ... the solution is of the class F1
%              = "F2" ... the solution is of the class F2
%              = "F3" ... the solution is of the class F3
% * flag ... integer indicating whether the computed solution is a TLS or
%            non-generic solution
%        = "0" ... X is a classical TLS solution
%        = "i", i ≥ 1 ... X is a non-generic solution & singular vectors
%                         corresponding to i distinct singular values must
%                         be added to the subspace to construct the solution

%   Optional OUTPUT parameters
% * classes ... row vector of all classes (in each run of the while loop)
% * Vmin ... resulting transformed right singular subspace
% * S ... singular values of [B,A] computed by "svd" function
% * V ... right singular space computed by "svd" function
% * left_multiplicities ... row vector of left multiplicities of all used
%                       singular values
% * right_multiplicities ... row vector of right multiplicities of all used
%                        singular values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X,orig_class,flag, classes,Vmin,S,V,left_multiplicities,right_multiplicities] = blockTLS(A,B,tol)

% Inicialization
m = size(A,1); n = size(A,2); d = size(B,2);
if m <= n + d
    Anew = zeros(n+d+1,n); Anew(1:m,:) =  A; A = Anew;
    Bnew = zeros(n+d+1,d); Bnew(1:m,:) =  B; B = Bnew;
end

flag = 0;
class = "S";
classes = [];
current_index = n+1;
left_multiplicities = [];
right_multiplicities = [];

% Construction of the right singular subspace
[~,S,V] = svd([B,A],"vector");


while (class == "S") | (class == "F3")  % until a solution is found

    if current_index == 0
        warning("The whole right singular subspace was searched and no solution was found.")
        X=-1; Vmin = "None";
        break
    end

    % Determining the left multiplicity of the current singular value
    lelf_multiplicity = 0;
    if current_index > 1
        l = current_index - 1;
        while ((S(l)-S(current_index))/S(current_index)) < tol 
            lelf_multiplicity = lelf_multiplicity + 1;
            l = l - 1;
            if l == 0
                break
            end
        end
    end
    left_multiplicities = [left_multiplicities, lelf_multiplicity];

    
    % Determining the right multiplicity of the (n+1)th singular value
    right_multiplicity = 1;             % including current singular value
    if flag == 0                        % not for non-generic solutions
        r = n+2;
        while abs((S(r)-S(n+1))/S(n+1)) < tol 
            right_multiplicity = right_multiplicity + 1;
            r = r + 1;
            if r == n + d + 1
                break
            end
        end
    end
    right_multiplicities = [right_multiplicities, right_multiplicity];
    
    % Defining the current right singular subspace in which the solution is sought
    V12 = V(1:d,(current_index - lelf_multiplicity):(current_index+right_multiplicity-1));
    V13 = V(1:d,(current_index+right_multiplicity):end);

    % Solution class determination
    r23 = rank([V12,V13],tol);
    r2 = rank(V12,tol);
    r3 = rank(V13,tol);
    fullrank = min(size(V13));

    if r23 < d
        class = "S";
    elseif (r2 == right_multiplicity) && (r3 == fullrank)
        class = "F1";
    elseif (r2 > right_multiplicity) && (r3 == fullrank)
        class = "F2";
    elseif (r2 > right_multiplicity) && (r3 < fullrank)
        class = "F3";
    end
    
    if flag == 0
        orig_class = class;
    end
    classes = [classes,class];
    
    % Solution construction or generalization to non-generic solution
    if (class == "S") | (class == "F3") % if a solution can't be constructed,
                                        % enlarge the right singular subspace
        flag = flag + 1;
        current_index = current_index - lelf_multiplicity - 1;
    else                                % Householder reflection to Vmin to
                                        % construct a solution minimal in norm
        Vmin = V(:,current_index-lelf_multiplicity:end);
        dim = size(Vmin,2);
        for i = 1:d
            v = Vmin(i,(1:end-i+1))';
            e = zeros(dim-i+1,1); e(end) = 1;
            if sign(v(end)) ~= 0
                q = v + sign(v(end)) * norm(v) * e;
            else
                q = v + norm(v) * e;
            end
            q = q / norm(q);
            h = eye(dim-i+1) - 2*(q*q');
            H = zeros(dim);
            H(1:(end-i+1),1:(end-i+1)) = h;
            if i > 1
                H((end-i+2):end,(end-i+2):end) = eye(i-1);
            end
            Vmin = Vmin * H;
            % zero out entries that should have been eliminated by the
            % Householder reflection, but, due to FPA, are not exactly zero
            % (instead, they are below the machine epsilon threshold)
            Vmin(i,(1:end-i)) = zeros(1,dim-i);
        end
        X = (- Vmin((d+1):end,(end-d+1):end)) / Vmin(1:d,(end-d+1):end);
    end
end
