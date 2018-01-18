% puzzle3.lp  -- Martian-Venusian Club, part 2
% author: Tommi Syrjänen 
%
% Raymond Smullyan presents the following puzzle in his book `Forever
% Undecided'.
%
% On Ganymede -- a satellite of Jupiter -- there is a club known as
% the Martian-Venusian Club. All members are either from Mars or from
% Venus, although visitors are sometimes allowed. An earthling is
% unable to distinguish Martians from Venetians by their appearance.
% Also, earthlings cannot distinguish either Martian or Venusian males
% from females, since they dress alike. Logicians, however, have an
% advantage, since the Venusian women always tell the truth and the
% Venusian men always lie. The martians are the opposite; the Martian
% men tell the truth and the Martian women always lie.
%
% The Martians and the Venetians often intermarry, and there are
% several mixed couples in the club. One couple approached the
% visitor and the following conversation ensued:
%
% Visitor: Where are you from?
% A: From Mars.
% B: That's not true!
%
% Was the couple mixed or not?
%

% All persons are from Mars or Venus
1 { martian(P), venetian(P) } 1 :- person(P).

% All persons are male or female
1 { female(P), male(P) } 1 :- person(P).

% All persons either lie or tell the truth depending on their origins
% and sex.
lies(P) :- person(P), martian(P), female(P).
lies(P) :- person(P), venetian(P), male(P).
truthful(P) :- person(P), martian(P), male(P).
truthful(P) :- person(P), venetian(P), female(P).

% A person may not tell the truth and lie at the same time
 :- person(P), lies(P), truthful(P).

% Persons:
person( a; b ).

% The persons in this puzzle are married so they can't have the same
% sex.
 :- male(a), male(b).
 :- female(a), female(b).

%% The hints:

% 1.
martian(a) :- truthful(a).
 :- martian(a), lies(a).

% 2.
lies(a) :- truthful(b).
 :- lies(a), lies(b).

