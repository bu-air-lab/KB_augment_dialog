% This planning problem has a solution with 11 steps
%
% Possible command line:
%   lparse -d none -c length=11 logistics.lp plan2.lp  | smodels 

% domain definitions:
object(package1).
object(package2).
object(package3).
object(package4).
object(package5).
object(package6).
object(package7).
object(package8).
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
airport(bos_airport).
location(bos_airport).
airport(pgh_airport).
location(pgh_airport).
airport(la_airport).
location(la_airport).
in_city(pgh_po, pgh).
in_city(pgh_airport, pgh).
in_city(bos_po, bos).
in_city(bos_airport, bos).
in_city(la_po, la).
in_city(la_airport, la).

% initial state 
at(package1, pgh_po, 1).
at(package2, pgh_po, 1).
at(package3, pgh_po, 1).
at(package4, pgh_po, 1).
at(package5, bos_po, 1).
at(package6, bos_po, 1).
at(package7, bos_po, 1).
at(package8, la_po, 1).
at(airplane1, pgh_airport, 1).
at(airplane2, pgh_airport, 1).
at(bos_truck, bos_po, 1).
at(pgh_truck, pgh_po, 1).
at(la_truck, la_po, 1).

%goal state:
compute models { 
at(package1, bos_po, length+1),
at(package2, bos_airport, length+1),
at(package3, la_po, length+1),
at(package4, la_airport, length+1),
at(package5, pgh_po, length+1),
at(package6, pgh_airport, length+1),
at(package7, pgh_po, length+1),
at(package8, pgh_po, length+1)
}.
