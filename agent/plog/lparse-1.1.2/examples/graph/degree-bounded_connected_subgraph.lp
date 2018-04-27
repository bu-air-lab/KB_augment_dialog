%% Problem: [GT26d] Directed Degree-Bounded_Connected_Subgraph
%% Input:
%% Directed Graph G = (V, A),
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%% Non-negative integer d <= |V|
%%     - command line constant
%% positive integer k <= |E|
%%     - command line constant
%%
%% Output:
%% A subset A' \subseteq A with |A| >= k such that the subgraph G' =
%% (V, A') is connected and has no vertex with degree exceeding d. 

%% Output predicate: in(X, Y): Arc (X, Y) is in A'

% Select at least k arcs:
k { in(X, Y) : arc(X, Y) }.

% A graph is connected if every vertex is reachable from every other vertex.
r(X, Y) :- arc(X, Y), in(X, Y).
r(Y, X) :- arc(X, Y), in(X, Y).
r(X, Z) :- r(X, Y), vtx(X), arc(Y, Z), 	in(Y, Z).
r(X, Z) :- r(X, Y), vtx(X), arc(Z, Y), 	in(Z, Y).

 :- vtx(X), vtx(Y), X < Y, not r(X, Y).

% Check that the degree limit is not exceeded:
 :- d + 1 { in(X, Y) : arc(X, Y) }, vtx(X).
