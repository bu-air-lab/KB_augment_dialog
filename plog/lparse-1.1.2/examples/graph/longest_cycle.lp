%% Problem: [ND28d] Directed Longest_Cycle
%% Input:
%% Directed Graph G = (V, A) and a length l(a) for  every a \in A,
%%      - vtx(X):       X is a vertex
%%      - arc(X, Y, L): (X, Y) is an arc with the length L
%% Positive integer k 
%%     - command line constant
%%
%% Output:
%% A simple cycle (v_1, ..., v_n) such that the sum of lengths of
%% arcs occurring in the cycle is at least k. 

%% Output predicate: lc(X, Y) : Arc (X, Y) is in the cycle
{ lc(X, Y) : arc(X, Y, L) } 1 :- vtx(X).

% The sum of the lengths must exceed the limit:
ok :- k [ lc(X, Y) : arc(X, Y, L) = L ] .
 :- not ok. 

% Next, we examine the vertices that occur in the cycle and choose the
% first of them to be the initial vertex. 
occurs(X) :- lc(X, Y), arc(X, Y, L).
initial(X) :- occurs(X), not occurs(Y) : vtx(X) : Y < X, vtx(X).

% Finally, every vertex that occurs in the cycle must be reachable
% from the initial vertex:
r(Y) :- lc(X, Y), initial(X), arc(X, Y, L).
r(Y) :- lc(X, Y), r(X), arc(X, Y, L).

 :- vtx(X), occurs(X), not r(X).
