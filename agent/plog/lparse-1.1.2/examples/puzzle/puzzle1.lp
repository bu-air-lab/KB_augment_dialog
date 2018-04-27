% puzzle1.lp  -- Knights and Knaves
% author: Tommi Syrjänen 
%
% Raymond Smullyan presents the following puzzle in his book `Forever
% Undecided'.
%
% The Island of Knights and Knaves has two types of inhabitants:
% knights, who always tell the truth, and knaves, who always lie.
%
% One day, three inhabitants (A, B, and C) of the island met a foreign
% tourist and gave the following information about themselves:
%
% 1. A said that B and C are both knights.
% 2. B said that A is a knave and C is a knight.
%
% What types are A, B, and C?
%

% There are three persons in this puzzle
person(a; b; c).

% Each person is either knight or knave, but not both. 
1 { knight(P), knave(P) } 1 :- person(P).

% Rest of the rules model the hints.

%% Hint 1:

% If A tells the truth, both B and C are knights
2 { knight(b), knight(c) } 2 :- knight(a).

% If A lies, it is not possible that they are both knights 
  :- knave(a), knight(b), knight(c).

%% Hint 2:

% If B tells the truth, A is a knave and C is a knight 
2 { knave(a), knight(c) } 2 :- knight(b).

% If B lies, it is not possible that A is knave and C is knight
  :- knave(b), knave(a), knight(c).

