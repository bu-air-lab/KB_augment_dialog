% This planning problem has a solution with 9 steps 

% Possible command line:
%   lparse -d none -c length=9 logistics.lp plan1.lp  | smodels 


% domain definitions:
object(package1).
object(package2).
object(package3).
plane(airplane1).
plane(airplane2).
city(pgh).
city(bos).
city(la).
truck(pgh_truck).
truck(bos_truck).
truck(la_truck).
location(bos_po).
location(la_po).
location(pgh_po).
location(bos_central).
location(la_central).
location(pgh_central).
airport(bos_airport).
location(bos_airport).
airport(pgh_airport).
location(pgh_airport).
airport(la_airport).
location(la_airport).
in_city(pgh_po, pgh).
in_city(pgh_airport, pgh).
in_city(pgh_central, pgh).
in_city(bos_po, bos).
in_city(bos_airport, bos).
in_city(bos_central, bos).
in_city(la_po, la).
in_city(la_airport, la).
in_city(la_central, la).

% initial state 
at(package1, pgh_po, 1).
at(package2, pgh_po, 1).
at(package3, pgh_po, 1).
at(airplane1, pgh_airport, 1).
at(airplane2, pgh_airport, 1).
at(bos_truck, bos_po, 1).
at(pgh_truck, pgh_po, 1).
at(la_truck, la_po, 1).

% goal state:
compute models { 
at(package1, bos_po, length+1),
at(package2, la_po, length+1),
at(package3, bos_po, length+1)
}.
