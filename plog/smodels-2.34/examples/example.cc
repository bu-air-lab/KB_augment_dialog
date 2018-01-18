// Compute the stable models of the program
//    a :- not b.
//    b :- not a.
#include <iostream>
#include "smodels.h"
#include "api.h"
#include "atomrule.h"

using namespace std;

int main ()
{
  Smodels smodels;
  Api api (&smodels.program);

  // You'll have to keep track of the atoms not remembered yourself
  api.remember ();

  Atom *a = api.new_atom ();
  Atom *b = api.new_atom ();
  api.set_name (a, "a");      // You can give the atoms names.
  api.set_name (b, "b");

  api.begin_rule (BASICRULE);
  api.add_head (a);
  api.add_body (b, false);  // Add "not b" to the body.
  api.end_rule ();
  api.begin_rule (BASICRULE);
  api.add_head (b);
  api.add_body (a, false);
  api.end_rule ();

  // You would add the compute statement here, e.g.,
  // api.set_compute (a, true) demands that a is in the model.

  api.done ();  // After this you shouldn't change the rules.

  smodels.program.print ();  // You can display the program.

  smodels.init ();  // Must be called before computing any models.

  // Compute all stable models.
  while (smodels.model ())  // Returns 0 when there are no more models
    smodels.printAnswer ();  // Prints the answer

  // Of course, you can inspect the atoms directly.

  smodels.revert (true);  // Forget everything that happened after init ().

  b->computeFalse = true;  // compute { not b }
  // Alternatively, api.set_compute (b, false).
  // api.reset_compute (Atom *, bool) removes atoms from the compute
  // statement.

  if (smodels.model ())  // There is a model.
    {
      Atom *c = api.get_atom ("a");
      if (c->Bpos)
	cout << c->atom_name () << " is in the stable model" << endl;
      if (c->Bneg)
	cout << c->atom_name () << " is not in the stable model" << endl;
    }

  return 0;
}
