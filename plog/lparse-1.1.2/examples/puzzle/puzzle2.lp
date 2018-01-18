% puzzle2.lp  -- Martian-Venusian Club, part 1
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
% One day a visitor met two Club members, Ork and Bog, who made the
% following statements:
%
% 1. Ork: Bog is from Venus.
% 2. Bog: Ork is from Mars.
% 3. Ork: Bog is male.
% 4. Bog: Ork is female.
%
% Where are Ork and Bog from, and are they male or female?

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
person( ork; bog ).

%% Hints:

% 1.
venetian(bog) :- truthful(ork).
 :- lies(ork), venetian(bog).

% 2.
martian(ork) :- truthful(bog).
 :- lies(bog), martian(ork).

% 3.
male(bog) :- truthful(ork).
 :- lies(ork), male(bog).

% 4.
female(ork) :- truthful(bog).
 :- lies(bog), female(ork).
