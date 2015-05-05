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
]'; % transposed

% initial distribution Nx1
X0 = randi(1000, N, 1); % N x 1
X0 = X0/sum(X0); % random distribution

%% SOLVER
% states
Xc = sdpvar(N*H,1); % NH x 1
Xid = [0.2 * ones(n,1)/n; zeros(m,1); 0.8]; % N x 1

% inputs
U = sdpvar(N*H,N); % NH x N

% A cap
% Ac = sdpvar(N*H,N); % NH x N
% Ac(1:N,:) = A + U(1:N,:);
% for k=2:H
%     Ac((k-1)*N+1:k*N,:) = ...
%         Ac((k-2)*N+1:(k-1)*N,:) * (A+U((k-1)*N+1:k*N,:));
% end

% constraints
Constraints = [ ...
    %Xc == Ac * X0 ... % system dynamics
    sum(reshape(Xc,[N,H])) == ones(1,H) ... distribution constraint on states
    Xc <= 1 ... % probability upper bound
    Xc >= 0 ... % probability lower bound
    sum(reshape(U,[N H*N])) == zeros(1,N*H) ... % U zero-sum per columns
    repmat(A,[H 1]) + U <= 1 ... % upper bound
    repmat(A,[H 1]) + U >= 0 ... % lower bound
];

% system dynamics
Constraints = [ Constraints, Xc(1:N) == (A + U(1:N,:)) * X0 ];
for k=2:H
    Constraints = [ Constraints ...
        Xc((k-1)*N+1:k*N) == ...
            (A + U((k-1)*N+1:k*N,:)) * Xc((k-2)*N+1:(k-1)*N) ...
    ];
end
% weights matrixes
Wx = ones(N,1); % same weight for every node
Wu = rand(N); % random weights

% cap variables
Wxc = repmat(Wx,[H 1])'; % 1 x NH
Xidc = repmat(Xid,[H 1]); % NH x 1
Wuc = repmat(Wu,[H 1]); % NH x N

% performance index
Objective =  Wxc*(Xc - Xidc).^2 ;
%Objective =  Wxc*(Xc - Xidc).^2 + Wuc .* U.^2;

% run
%Options = sdpsettings('solver','bmibnb');
optimize(Constraints, Objective, []);

%% PLOT
% graphic plot
valU = value(U);
valX = value(Xc);

v = zeros(H,N);
hold on;
i=1;
for j=1:N
    for k=1:H
        v(k,j) = valU((k-1)*N+i,j);
    end
end
h = plot(v);

set(h(1:n),'Color','b'); % standard nodes
set(h(n+1:n+m),'Color','r'); % corrupted nodes
set(h(n+m+1),'Color','g'); % receiver
grid on;
legend('1', '2', '3', '4', '5', '6', '7')
title('Parameters')

figure;
h = plot([X0'; reshape(valX, [N H])']);
set(h(1:n),'Color','b'); % standard nodes
set(h(n+1:n+m),'Color','r'); % corrupted nodes
set(h(n+m+1),'Color','g'); % receiver
grid on
legend('1', '2', '3', '4', '5', '6', '7')
title('System states')
