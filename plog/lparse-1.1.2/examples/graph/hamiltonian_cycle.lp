%% Problem: [GT38d] Directed_Hamiltonian_Cycle
%% Input:
%% Directed Graph G = (V, A).
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%%
%% Output:
%% A simple cycle that visits every vertex in G.

%% Output predicate:  hc(X, Y): Arc (X, Y) is in the cycle

% The Hamiltonian cycle visits every vertex exactly once so we need
% to select arc to leave every vertex:
{ hc(X, Y) : arc(X, Y) } 1 :- vtx(X).

r(Y) :- arc(X, Y), hc(X, Y), initial(X).
r(Y) :- arc(X, Y), hc(X, Y), r(X).

 :- vtx(X), not r(X).

