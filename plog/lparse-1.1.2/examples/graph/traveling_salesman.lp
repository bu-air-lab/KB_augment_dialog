% Problem: [ND22] Directed Traveling Salesman
%    - A weighted directed  graph G 
%          - vtx(X):        X is a vertex 
%          - arc(X, Y, W): (X, Y) is an arc with weight W
%    - A positive integer k
%          - command line constant
% 
% Output: 
%    - A Hamiltonian circuit of G such that the sum of the weights of
%      the arcs in it is at most k.
%        - hc(X, Y): Arc (X, Y) belongs to the cycle

% The Hamiltonian cycle visits every vertex exactly once so we need
% to select arc to leave every vertex:
{ hc(X, Y) : arc(X, Y, W) } 1 :- vtx(X).

% The cycle may not be too long
 :- k + 1 [ hc(X, Y) : arc(X, Y, W) ] .

% Every vertex must be reachable from an initial one:
initial(X) :- vtx(X), f : vtx(Y) : Y < X.

r(Y) :- arc(X, Y, W), hc(X, Y), initial(X).
r(Y) :- arc(X, Y, W), hc(X, Y), r(X).

 :- vtx(X), not r(X).
