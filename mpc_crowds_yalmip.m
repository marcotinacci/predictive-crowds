%% MODEL PARAMS
% safe nodes
n = 4;
% corrupted nodes
m = 2;
% forward probability
pf = 0.5;
% multiplying parameter factor 
fb = 1;
% controlled horizon
T = 10;
% number of parameters
M = 5;

%% DERIVED DATA
% total nodes
N = m+n+1;
% uniform choice probability
pu = pf / (m+n);
% uniform restart porobability
pr = 1 / N;

%% MODEL DYNAMICS
% DTMC stochastic (per columns) matrix NxN
A = [
    ones(m+n) * pu, ones(m+n,1) * (1-pf);
    ones(1,N) * pr
]'; % transposed!
% zero-sum matrix NxM
%B = zeros(N,M);%ones(N,M); 
%B = [ones(m+n,1) * -fb/(m+n); fb];
B = [rand(N-1,M)*2-1; zeros(1,M)];
B(N,:) = -sum(B); % zero-sum condition

C = eye(N);
% initial distribution Nx1
X0 = [ 1; zeros(N-1,1) ]; % dirac on the first state
%X0 = randi(1000, N, 1); X0 = X0/sum(X0); % random distribution
% weights
P = diag([ones(1,n) 10*ones(1,m) 1]); % NxN
Q = diag([ones(1,n) 10*ones(1,m) 1]); % NxN
R = 10*eye(M); % MxM

%% SOLVER
% states
X = sdpvar(N,T); % N x T
cX = reshape(X,[N*T,1]); % N*T x 1
% inputs
U = sdpvar(M,T); % M x T
cU = reshape(U,[M*T,1]); % M*T x 1

% build T bar and S bar matrixes
bT = zeros(N*T,N); % N*T x N
bS = zeros(N*T,M*T); % N*T x M*T
prevA = eye(N);
cumA = A;
for i=1:T
    bT(((i-1)*N)+1:i*N,:) = cumA;
    % i-th sub-diagonal
    pAB = prevA*B;
    for j=1:T-i+1
        bS(((j-1+i-1)*N)+1:(j+i-1)*N,((j-1)*M)+1:j*M) = pAB;
    end
    prevA = cumA;
    cumA = cumA * A;
end

% constraints
Constraints = [ ...
    cX == bS*cU + bT*X0 ... system dynamics
    sum(X) == ones(1,T) ... distribution constraint on states
    X <= 1 ...
    X >= 0 ...
    U <= 1 ...
    U >= -1 ...
    ];

% build Q bar and R bar matrixes
bQ = zeros(N*T);
bR = zeros(M*T);
for i=1:T-1
    bQ((i-1)*N+1:i*N,(i-1)*N+1:i*N) = Q;
    bR((i-1)*M+1:i*M,(i-1)*M+1:i*M) = R;
end
bQ((T-1)*N+1:T*N,(T-1)*N+1:T*N) = P; % last matrix of the diagonal
bR((T-1)*M+1:T*M,(T-1)*M+1:T*M) = R;

% performance index
Objective = X0' * Q * X0 + cX' * bQ * cX + cU' * bR * cU;

% run
optimize(Constraints, Objective, []);

%% PLOT
% raw data
value(X)
value(U)
% graphic plot
subplot(211)
h = plot([X0'; value(X')]);
set(h(1:n),'Color','b'); % standard nodes
set(h(n+1:n+m),'Color','r'); % corrupted nodes
set(h(n+m+1),'Color','g'); % receiver
grid on
legend('1', '2', '3', '4', '5', '6', '7')
title('System states')
subplot(212)
plot([zeros(1,M); value(U')])
grid on
legend('1', '2', '3', '4', '5', '6', '7')
title('Control action')
