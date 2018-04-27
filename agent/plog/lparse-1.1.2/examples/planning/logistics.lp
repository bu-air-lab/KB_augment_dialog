% logistics.lp --- smodels operator file for logistic domain
%
% Author: Tommi Syrjänen
%
% This program encodes a logistics planning domain as smodels
% program. In the domain there are packets that must be delivered from
% one place to another. The packages can be transported by trucks and
% airplanes. Airplanes can only fly between airports and trucks may
% not leave the city they start in.
%
% The possible planning actions are:
%    load_truck(Object, Truck, Time)
%    unload_truck(Object, Truck, Time)
%    drive_truck(Truck, From, To, Time)
%    load_airplane(Object, Plane, Time)
%    unload_airplane(Object, Plane, Time)
%    fly_airplane(Plane, From, To, Time)
%
% The idea of this translation is to encode actions using choice
% rules, in form:
%
% 	{ action } :- preconditions.
%
%
% According to the semantics of choice rules, if the body is true in a
% model then the head may be true in the model, but it doesn't have to
% be true in it.
%
% The effects of an action are encoded in a normal way:
%
%	effects :- action.
%
% The blocking of conflicting actions is done by adding a constraint
% that says that if an action doesn't change its own precondition,
% then the precondition has to hold also at the next instant:
%
%      precondition(I+1) :- action, precondition(I),
%                           keeps_precondition(action).
%
% 
% Usage:
% 	lparse -d none -c lenght=n logistics.lp fact_file.lp | smodels 
%
% The fact_file should contain the description of the initial and goal
% states as well as definitions of the domain predicates. 
%
% This logic program needs lparse version 0.99.7 or newer and smodels
% version 2.x or newer. 
%
% Uncomment the following line if you want to set the lenght of the
% plan manually without having to use command line arguments. 
%const lenght = 10.

% Define the number of models:
const models = 1.

% Define the time steps:
time(1..length).


%% ACTIONS

% Action:load_truck(Obj, Tr, I)
% Preconditions: at(Obj,Loc, I) & at(Tr, Loc, I)
% Postconditions: in(Obj, Tr, I+1)

{ load_truck(Obj, Tr, I) } :-
	at(Obj, Loc, I),
	at(Tr, Loc, I),
	object(Obj),
	truck(Tr),
	location(Loc),
	time(I).

% Effects:

in(Obj, Tr, I+1) :-
	load_truck(Obj, Tr, I),
	truck(Tr),
	object(Obj),
	time(I).


changes(Obj, I) :-
	load_truck(Obj, Tr, I),
	truck(Tr),
	object(Obj),
	time(I).


% As one of the preconditions for load_truck(Obj, Tr, I) is
% at(Tr, Loc, I) and the operator doesn't change it, the truck has to
% be at the same place at the next instant. 

at(Tr, Loc, I+1) :-
	load_truck(Obj, Tr, I),
	at(Tr, Loc, I),
	location(Loc),
	truck(Tr),
	object(Obj),
	time(I).
	
% Action: load_airplane(Obj, Pl, I)
% Preconditions: at(Obj,Loc, I) & at(Pl, Loc, I)
% Postconditions: in(Obj, Pl, I+1)

{ load_airplane(Obj, Pl, I) } :-
	at(Obj, Loc, I),
	at(Pl, Loc, I),
	object(Obj),
	plane(Pl),
	airport(Loc),
	time(I).

% Effects:

in(Obj, Pl, I+1) :-
	load_airplane(Obj, Pl, I),
	plane(Pl),
	object(Obj),
	time(I).

at(Pl, Loc, I+1) :-
	load_airplane(Obj, Pl, I),
	at(Pl, Loc, I),
	airport(Loc),
	plane(Pl),
	object(Obj),
	time(I).

changes(Obj, I) :-
	load_airplane(Obj, Pl, I),
	plane(Pl),
	object(Obj),
	time(I).

% Action: unload_truck(Obj, Tr, I)
% Preconditions: in(Obj, Tr, I) & at(Tr, Loc, I)
% Postconditions: at(Obj, Loc, I+1)

{ unload_truck(Obj, Tr, I) } :-
	in(Obj, Tr, I),
	object(Obj),
	truck(Tr),
	time(I).

% Effects:

at(Obj, Loc, I+1) :-
	unload_truck(Obj, Tr, I),
	at(Tr, Loc, I),
	location(Loc),
	object(Obj),
	truck(Tr),
	time(I).

at(Tr, Loc, I+1) :-
	unload_truck(Obj, Tr, I),
	at(Tr, Loc, I),
	location(Loc),
	truck(Tr),
	object(Obj),
	time(I).

changes(Obj, I) :-
	unload_truck(Obj, Tr, I),
	truck(Tr),
	object(Obj),
	time(I).

% Action: unload_plane(Obj, Pl, I)
% Preconditions: in(Obj, Pl, I)
% Postconditions: at(Obj, Loc, I+1)

{ unload_airplane(Obj, Pl, I) } :-
	in(Obj, Pl, I),
	object(Obj),
	plane(Pl),
	time(I).

% Effects:

at(Pl, Loc, I+1) :-
	unload_airplane(Obj, Pl, I),
	at(Pl, Loc, I),
	plane(Pl),
	object(Obj),
	airport(Loc),
	time(I).

at(Obj, Loc, I+1) :-
	unload_airplane(Obj, Pl, I),
	at(Pl, Loc, I),
	airport(Loc),
	object(Obj),
	plane(Pl),
	time(I).

changes(Obj, I) :-
	unload_airplane(Obj, Pl, I),
	plane(Pl),
	object(Obj),
	time(I).

% Action: drive_truck(Tr, From, To, I)
% Precondition: at(Tr, From, I)
% Postcondition: at(Tr, To, I+1)

{ drive_truck(Tr, From, To, I) } :-
	at(Tr, From, I),
	truck(Tr),
	same_city(From, To),
	From != To,
	time(I).

% Effects:

at(Tr, To, I+1) :-
	drive_truck(Tr, From, To, I),
	truck(Tr),
	same_city(From, To),
	From != To,
	time(I).

changes(Tr, I) :-
	drive_truck(Tr, From, To, I),
	truck(Tr),
	same_city(From, To),
	From != To,
	time(I).

% Action: fly_airplane(Pl, From, To, I)
% Preconditions: at(Pl, From, I)
% Postconditions: at(Pl, To, I)

{ fly_airplane(Pl, From, To, I) } :-
	at(Pl, From, I),
	plane(Pl),
	airport(From),
	airport(To),
	From != To,
	time(I).

% Effects:

at(Pl, To, I+1) :-
	fly_airplane(Pl, From, To, I),
	plane(Pl),
	airport(From),
	airport(To),
	From != To,
	time(I).

changes(Pl, I) :-
	fly_airplane(Pl, From, To, I),
	plane(Pl),
	airport(From),
	airport(To),
	From != To,
	time(I).


%% FRAME AXIOMS

% Everything stays at the same place where it is unless some action
% moves it.

at(Obj, Loc, I+1) :-
	at(Obj, Loc, I),
	not changes(Obj, I),
	object(Obj),
	location(Loc),
	time(I).

in(Obj, Cont, I+1) :-
	in(Obj, Cont, I),
	not changes(Obj, I),
	object(Obj),
	container(Cont),
	time(I).

at(Cont, Loc, I+1) :-
	at(Cont, Loc, I),
	not changes(Cont, I),
	container(Cont),
	location(Loc),
	time(I).

% An object may not be in two places at the same time

 :- 2 { at(Obj, Loc, I) : location(Loc),
	in(Obj, Cont, I) : container(Cont) },
	object(Obj),
	end_time(I).

 :- 2 { at(Cont, Loc, I) : location(Loc) },
	container(Cont),
	end_time(I).

% Both trucks and airplanes are containers
container(X) :- truck(X).
container(X) :- plane(X).

same_city(Loc1, Loc2) :-
	in_city(Loc1, C),
	in_city(Loc2, C).

% end_time is used so that there is no need to write new constraint
% rule for the last time step 
end_time(I) :- time(I).
end_time(length+1).
