// Compute the stable models of the program
//    a :- not b.
//    b :- not a.
// Then add the rule
//    a :- b.
// and compute the stable models of the resulting program.
#include <iostream>
#include "smodels.h"
#include "api.h"
#include "atomrule.h"

int main ()
{
  Smodels smodels;
  Api api (&smodels.program);

  Atom *a = api.new_atom ();
  Atom *b = api.new_atom ();
  api.set_name (a, "a");
  api.set_name (b, "b");
  // a :- not b.
  api.begin_rule (BASICRULE);
  api.add_head (a);
  api.add_body (b, false);
  api.end_rule ();
  // b :- not a.
  api.begin_rule (BASICRULE);
  api.add_head (b);
  api.add_body (a, false);
  api.end_rule ();
  // Copy the program
  Smodels smodels2;
  Api api2 (&smodels2.program);
  api2.copy (&api);
  api2.done ();
  // a :- b.
  api.begin_rule (BASICRULE);
  api.add_head (a);
  api.add_body (b, true);
  api.end_rule ();
  api.done ();

  // Find the models of the smaller program
  smodels2.program.print ();
  smodels2.init ();
  while (smodels2.model ())
    smodels2.printAnswer ();

  // Find the models of the larger program
  smodels.program.print ();
  smodels.init ();
  while (smodels.model ())
    smodels.printAnswer ();

  return 0;
}
