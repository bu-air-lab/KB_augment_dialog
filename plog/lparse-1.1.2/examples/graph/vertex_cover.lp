%% Problem: [GT1d] Directed Vertex_Cover
%% Input:
%% Directed Graph G = (V, A)
%%     - vtx(X):   X is a vertex
%%     - arc(X, Y):  (X, Y) is an arc
%%  positive integer k <= |V|.
%%     - command line constant

%% Output:
%% A subset V' \subseteq V such that |V'| <= k and for all (u, v) \in
%% A, u \in V' or v \in V'.

%% Output predicate: in(X):  Vertex X is in the cover

% First, choose at most k vertices:
{ in(X) : vtx(X) } k.

% Then, it is an error if an arc is not covered
 :- arc(X, Y), not in(X), not in(Y).
