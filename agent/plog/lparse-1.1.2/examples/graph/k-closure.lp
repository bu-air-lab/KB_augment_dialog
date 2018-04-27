%% Problem: [GT58d] Directed K-Closure
%% Input:
%% Directed graph G = (G, A)
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%% Positive integer k <= |V|
%%     - command line constant
%% Output:
%% A subset V' \subseteq V with |V'| <= k such that for all (u, v) \in
%% A it holds that  u \in V' or v \notin V'.  

%% Output predicate: in(X):  Vertex X \in V'

% Select at most k vertices:
{ in(X) : vtx(X) }  k.

% Check that all arcs are closed:
ok(X, Y) :-
	arc(X, Y),
	in(X).

ok(X, Y) :-
	arc(X, Y),
	not in(Y).

 :- arc(X, Y), not ok(X, Y).

