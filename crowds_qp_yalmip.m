%% MODEL PARAMS
% safe nodes
n = 4;
% corrupted nodes
m = 2;
% forward probability
pf = 0.5;
% prediction horizon
H = 10;

%% DERIVED DATA
% total nodes
N = m+n+1;
% uniform choice probability
pu = pf / (m+n);
% uniform restart probability
pr = 1 / N;

%% MODEL DYNAMICS
% DTMC stochastic (per columns) matrix NxN
A = [
    ones(m+n) * pu, ones(m+n,1) * (1-pf);
    ones(1,N) * pr
]'; % transposed!

% initial distribution Nx1
% X0 = [ 1; zeros(N-1,1) ]; % dirac on the first state
X0 = randi(1000, N, 1); % N x 1
X0 = X0/sum(X0); % random distribution

%% SOLVER
% states
X = sdpvar(H,N,1); % H x N x 1
cX = reshape(X,[N*H,1]); % N*H x 1
Xid = [0.2*ones(n,1)/n; zeros(m,1); 0.8]; % N x 1
%Xid = [ones(n,1); zeros(m,1); 1]; % N x 1
%Xid = Xid/sum(Xid); % ideal distribution

% dynamics
cA = zeros(H*N,N); % H*N x N
prevA = A; % N x N
for i=1:H
    cA((i-1)*N+1:i*N,:) = prevA;
    prevA = prevA * A;
end

% inputs
U = sdpvar(H,N,N); % H x N x N
%cU = reshape(U,N*H,N)'; % H*N x N
cU = sdpvar(H*N,N); % H*N x N
sumU = sdpvar(H*N,1); % H*N x 1
for k=1:H
    for j=1:N
        sumU((k-1)*N+j) = sum(cU((k-1)*N+1:k*N,j));
    end
end
% parameter boundaries
Ulb = repmat(-A,H,1);
Uub = repmat(ones(N)-A,H,1);

% constraints
Constraints = [ ...
    %sumU == zeros(H*N,1) ... sum-zero params (10)
    cX == cA*X0 + cU*ones(N,1) ... system dynamics
    cU <= Uub ... parameter upper bounds
    cU >= Ulb ... parameter lower bounds
    sum(X,2) == ones(H,1) ... distribution constraint on states
    X <= 1 ... probability upper bound
    X >= 0 ... probability lower bound
];

% weights matrixes
Wx = ones(N,1); % same weight for every node
%Wx = [ones(n,1); ones(m,1); 1]; % N x 1
Wu = rand(N);
% performance index
Obj = 0;
for i=1:N
    for k=1:H
        Obj = Obj + Wx(i)*(X(k,i)-Xid(i))^2;
        for j=1:N
            Obj = Obj + Wu(i,j) * cU((k-1)*N+i,j)^2;
        end
    end
end

% run
optimize(Constraints, Obj, []);

%% PLOT
% raw data
value(X)
value(cU)
% graphic plot
val = value(cU);
v = zeros(H,N);
hold on;
i=2;
for j=1:N
    for k=1:H
        v(k,j) = val((k-1)*N+i,j);
    end
end
h = plot(v);
grid on;
title('Parameters')

figure;
h = plot([X0'; value(X)]);
set(h(1:n),'Color','b'); % standard nodes
set(h(n+1:n+m),'Color','r'); % corrupted nodes
set(h(n+m+1),'Color','g'); % receiver
grid on
legend('1', '2', '3', '4', '5', '6', '7')
title('System states')
