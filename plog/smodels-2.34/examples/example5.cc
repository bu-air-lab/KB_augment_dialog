// Push/pop interface for smodels.
//
// An atom that is pushed behaves as if it was set in the compute
// statement, i.e., model() will not backtrack passed it.
//
// bool push (Atom *a, bool set_true)
//   Returns false if the atom is already set. Causes a conflict if
//   the atom is set to the opposite value of set_true. Also calls
//   expand(), which can cause a conflict.
//   The function push should not be called if conflict() returns
//   true. Observe that conflict() resets the conflict_found flag and
//   therefore only returns true once.
//
// bool pop (Atom *a)
//   Returns true if the atom is successfully popped. One can only pop
//   atoms that have guess == true, i.e., atoms that have been pushed
//   or chosen by the heuristic. When an atom is popped smodels
//   backtracks until the atom has been removed from the stack.
//
#include <iostream>
#include "print.h"
#include "atomrule.h"
#include "api.h"
#include "smodels.h"
#include "read.h"

using namespace std;

class SmodelsPop : public Smodels
{
public:
  SmodelsPop ();

  bool pop (Atom *);
  bool push (Atom *, bool);

  void setup ();
  void backtrack ();
  bool model (); // Searches for a model, returns true if it finds one

protected:
  bool do_backtrack;
};

SmodelsPop::SmodelsPop ()
{
  do_backtrack = false;
}

bool
SmodelsPop::pop (Atom *a)
{
  if (a->guess == false)
    return false; // Shouldn't call pop on these
  do_backtrack = false;
  fail = false;
  conflict_found = false;
  Atom *b = 0;
  while (stack.top > setup_top)
    {
      b = stack.pop ();
      if (b->guess)
	{
	  b->computeTrue = false;
	  b->computeFalse = false;
	  b->guess = false;
	  guesses--;
	}
      PRINT_STACK (b->backtracked = false);
      PRINT_STACK (b->forced = false);
      if (b->Bpos)
	b->backtrackFromBTrue ();
      else if (b->Bneg)
	b->backtrackFromBFalse ();
      if (a == b)
	break;
    }
  atomtop = program.number_of_atoms;
  return a == b;
}

bool
SmodelsPop::push (Atom *a, bool set_true)
{
  if (conflict_found)
    return false; // Can't push if we have a conflict
  do_backtrack = false;
  if (set_true == false && a->Bneg == false)
    {
      if (a->Bpos)
	{
	  set_conflict ();
	  return false;
	}
      else
	{
	  a->computeFalse = true;
	  a->guess = true;
	  guesses++;
	  setToBFalse (a);
	  expand ();
	  return true;
	}
    }
  else if (set_true && a->Bpos == false)
    {
      if (a->Bneg)
	{
	  set_conflict ();
	  return false;
	}
      else
	{
	  a->computeTrue = true;
	  a->guess = true;
	  guesses++;
	  setToBTrue (a);
	  expand ();
	  return true;
	}
    }
  return false;
}

void
SmodelsPop::setup ()
{
  setup_with_lookahead ();
  if (!conflict () && !covered ())
    {
      heuristic ();
      improve (true);
    }
}

void
SmodelsPop::backtrack ()
{
  if (guesses == 0)
    {
      fail = true;
      return;
    }
  Atom *a = unwind (true);
  if (a->computeTrue || a->computeFalse)
    {
      a->guess = true;
      guesses++;
      stack.push (a);
      addAtom (a);
      removeAtom (atomtop-1);
      fail = true;
      return;
    }
  PRINT_STACK (a->backtracked = true);
  if (a->Bneg)
    {
      a->backtrackFromBFalse ();
      a->setBTrue ();
    }
  else
    {
      a->backtrackFromBTrue ();
      a->setBFalse ();
    }
  stack.push (a);
  addAtom (a);
  number_of_wrong_choices++;
}

bool
SmodelsPop::model ()
{
  if (do_backtrack)
    backtrack ();
  while (!fail)
    {
      expand ();
      if (conflict ())
	backtrack ();
      else if (covered ())
	{
	  answer_number++;
	  program.set_optimum ();
	  do_backtrack = true;
	  return true;
	}
      else
	lookahead ();
    }
  number_of_wrong_choices--;
  do_backtrack = false;
  return false;
}


// Example of use. hamilton -nodes 10 -new | pparse | ./example5

int main (int, char *[])
{
  SmodelsPop smodels;
  Api api (&smodels.program);
  api.remember (); // We want to look up atoms using their names
  Read reader (&smodels.program, &api);
  if (reader.read (cin))
    return 1;  // Error in input
  api.done ();
  smodels.init ();

  // Atoms in the compute statement are fixed and can not be changed
  // For example the atom "false" in this example
  smodels.setup (); // Do the setup including optimizations and lookahead
  Atom *a = api.get_atom ("v2v6"); // Push an atom
  smodels.push (a, true);
  if (smodels.conflict ())
    cout << "Not found" << endl;
  else
    {
      Atom *b = api.get_atom ("v3v8"); // Push an atom
      smodels.push (b, false);
      if (smodels.conflict ())
	cout << "Not found" << endl;
      else
	// Compute a model
	if (smodels.model ())
	  smodels.printAnswer ();
	else
	  cout << "Not found" << endl;
    }
  // Pop atom a, atom b is automatically popped.
  smodels.pop (a);
  // Push a again.
  smodels.push (a, false);
  // Compute a model
  if (smodels.model ())
    smodels.printAnswer ();
  return 0;
}
