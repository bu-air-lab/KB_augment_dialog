%% Problem:  Maxsat 
%%
%% Input: A set of propositional clauses
%%        - clause(C):    C is a clause
%%        - atom(A):      A is a propositional atom
%%        - pos(C, A):    A occurs in clause C
%%        - neg(C, A):    -A occurs in clause C
%% A positive integer k
%%        - command line constant
%%  
%% Output:
%% A truth assignment that satisfies at least k clauses. 
%%
%% Output predicate: true(A):  Atom A is true in the assignment

%% Choose an arbitrary truth value for each atom:
{ true(A) : atom(A) }.

%% A clause is satisfied if contains a satisfied literal:
sat(C) :- pos(C, A), true(A).
sat(C) :- neg(C, A), not true(A). 

% It is an error if less than k clauses are satisfied:
 :- {  sat(C) : clause(C) } k-1.

