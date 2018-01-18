%% Problem: [GT19d] Directed Clique
%% Input:
%% Directed Graph G = (V, A),
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%% Positive integer k <= |V|
%%     - command line constant
%%
%% Output:
%% A subset V' \subseteq V such that |V'| >= k and for all u, v \in V'
%% it holds that (u, v) \in V

%% Output predicate: in(X):  X is in clique

% Choose vertices into clique
k { in(X) : vtx(X) }.

% It is an error if there is no edge connecting two vertices in the
% clique:
 :- in(X), in(Y), not arc(X, Y), vtx(X), vtx(Y).
