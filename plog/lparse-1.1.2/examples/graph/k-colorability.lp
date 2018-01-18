%% Problem: [GT4d] Directed Graph k-Colorability
%% Input:
%% Directed Graph G = (V, A)
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%% Positive integer k <= |V|
%%     - command line constant

%% Output:
%% A function f : V -> { 1, ..., k } such that forall (u, v) \in A:
%% f(u) != f(v) 

%% Output predicate: has_color(X, C): Vertex X has color C.

% There are k colors:
color(1 .. k).

% Every vertex has a color:
1 { has_color(X, C) : color(C) } 1 :- vtx(X).

% Two adjacent vertices have different colors
 :- has_color(X, C),
	has_color(Y, C),
	arc(X, Y),
	color(C).
