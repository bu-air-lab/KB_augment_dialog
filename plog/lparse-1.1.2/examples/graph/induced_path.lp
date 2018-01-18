%% Problem: [GT23d] Directed Induced_Path
%% Input:
%% Directed Graph G = (V, A).
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%% Positive integer k <= |V|.
%%     - command line constant
%%
%% Output:
%% A subset V' \subseteq V with |V'| >= k such that the subgraph
%% induced by V' is a simple path.

%% Output predicate: in(X): Vertex X is in V'

% First, choose at least k vertices:
k { in(X) : vtx(X) } .

% Next, identify the arcs in the induced graph. 
arc_S(X, Y) :-
	arc(X, Y),
	in(X),
	in(Y).

% There must be exactly one initial vertex:
initial(Y) :-
	not arc_S(X, Y) : arc(X, Y),
	vtx(Y),
	in(Y).
 :- 2 { initial(X) : vtx(X) }.

% All vertices in subset must be reachable from it:
r(Y) :- initial(X), arc_S(X, Y), arc(X, Y).
r(Y) :- r(X), arc_S(X, Y), arc(X, Y).
 :- in(X), not initial(X), not r(X).

% Finally, it is an error if there are two arcs leading from any
% vertex in the subset:
 :- 2 { arc_S(X, Y) : arc(X, Y) }, vtx(X).
