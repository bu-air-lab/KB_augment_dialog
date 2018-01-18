%% Problem: [OPEN1d] Directed Graph_Isomorphism
%% Input:
%% Graph G = (V_1, A_1)
%%     - vtx_G(X):   X is a vertex
%%     - arc_G(X, Y):  (X, Y) is an arc
%%
%% Graph H = (V_2, A_2)
%%     - vtx_H(X):   X is a vertex
%%     - arc_H(X, Y):  (X, Y) is an arc
%%
%%
%% Output:
%%  A bijection f : V_1 -> V_2 such that (u, v) \in A_1 if and only if
%%    (f(u), f(v)) \in A_2.
%%
%% Output predicate:  maps_to(X, Y)    : f(X) = Y

%% Each vertex from V_1 maps to exactly one vertex from V_2:
1 { maps_to(X, U) : vtx_H(U) } 1 :- vtx_G(X).

%% Exactly one vertex from V_1 maps to each vertex in V_2:
1 { maps_to(X, U) : vtx_G(X) } 1 :- vtx_G(U).

%% An isomorphism preserves arcs between nodes.
 :- maps_to(X, U),
	maps_to(Y, V),
	arc_G(X, Y),
	not arc_H(U, V),
	vtx_H(U),
	vtx_H(V).

 :- maps_to(X, U),
	maps_to(Y, V),
	not arc_G(X, Y),
	arc_H(U, V),
	vtx_G(X),
	vtx_G(Y).


