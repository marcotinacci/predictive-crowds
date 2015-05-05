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
Xid = [0.2*ones(n,1)/n; zeros(m,1); 0.8]; % N x 1

% inputs
U = sdpvar(H,N,N); % H x N x N

% constraints
Constraints = [ ...
    U <= 1 ... parameter upper bounds (12)
    U >= -1 ... parameter lower bounds (12)
    sum(X,2) == ones(H,1) ... distribution constraint on states
    X <= 1 ... probability upper bound
    X >= 0 ... probability lower bound
];

% k = 1
for i=1:N
    temp = A(1,i)*X0(1) + U(k,1,i);
    temp2 = U(1,i,1);
    for j=2:N
        temp = temp + A(j,i)*X0(j) + U(k,j,i);
        temp2 = temp2 + U(1,i,j);
    end
    Constraints = [ Constraints, temp2 == 0, X(k,i) == temp ];
end
% k > 1
for k=2:H
    for i=1:N
        temp = A(1,i)*X(k-1,1) + U(k,1,i);
        temp2 = U(k,i,1);
        for j=2:N
            temp = temp + A(j,i)*X(k-1,j) + U(k,j,i);
            temp2 = temp2 + U(k,i,j);
        end
        Constraints = [ Constraints, temp2 == 0, X(k,i) == temp ];
    end
end

% weights matrixes
Wx = ones(N,1); % same weight for every node
Wu = rand(N); % random weights
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

set(h(1:n),'Color','b'); % standard nodes
set(h(n+1:n+m),'Color','r'); % corrupted nodes
set(h(n+m+1),'Color','g'); % receiver
grid on;
legend('1', '2', '3', '4', '5', '6', '7')
title('Parameters')

figure;
h = plot([X0'; value(X)]);
set(h(1:n),'Color','b'); % standard nodes
set(h(n+1:n+m),'Color','r'); % corrupted nodes
set(h(n+m+1),'Color','g'); % receiver
grid on
legend('1', '2', '3', '4', '5', '6', '7')
title('System states')
