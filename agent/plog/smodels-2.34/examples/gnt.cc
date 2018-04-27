// Copyright 1999 by Patrik Simons
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston,
// MA 02111-1307, USA.
//
// Patrik.Simons@hut.fi

// Compute answer sets for disjunctive logic programs.
// Before use change every disjunctive rule
//     a|b|..|c :- body.
// into the rule
//     {a,b,...,c} :- body.
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>
#include "atomrule.h"
#include "api.h"
#include "smodels.h"
#include "read.h"

using namespace std;

class AtomWithOther : public Atom
{
public:
  AtomWithOther (Program *p) : Atom (p) { other = 0; hasvalue = false; };
  AtomWithOther *other;
  bool hasvalue;
};

class ApiWithOther : public Api
{
public:
  ApiWithOther (Program *p) : Api (p) { no = 0; };
  Atom *new_atom ();        // Create new atom
  void add_constraints ();
  AtomWithOther *no;
};

Atom *
ApiWithOther::new_atom ()
{
  Atom *a = new AtomWithOther (program);
  program->atoms.push (a);
  program->number_of_atoms++;
  return a;
}

void
ApiWithOther::add_constraints ()
{
  no = static_cast<AtomWithOther *>(new_atom ());
  no->computeFalse = true;
  for (Node *n = program->rules.head (); n; n = n->next)
    if (n->rule->type == CHOICERULE)
      {
	ChoiceRule *r = static_cast<ChoiceRule *>(n->rule);
	Atom *body = 0;
	if (r->nbody != r->nend || r->pbody != r->pend)
	  {
	    body = new_atom ();
	    begin_rule (BASICRULE);
	    add_head (body);
	    for (Atom **a = r->nbody; a != r->nend; a++)
	      add_body (*a, false);
	    for (Atom **a = r->pbody; a != r->pend; a++)
	      add_body (*a, true);
	    end_rule ();
	  }
	begin_rule (BASICRULE);
	add_head (no);
	for (Atom **a = r->head; a != r->hend; a++)
	  add_body (*a, false);
	if (body)
	  add_body (body, true);
	end_rule ();
      }
}

class SmodelsAux : public Smodels
{
public:
  void setup ();
  bool model ();
};

void
SmodelsAux::setup ()
{
  Smodels::setup (true);
  for (Node *n = program.atoms.head (); n; n = n->next)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(n->atom);
      if (a->Bpos || a->Bneg)
	a->hasvalue = true;
    }
}

bool
SmodelsAux::model ()
{
  while (!fail)
    {
      expand ();
      if (conflict ())
	backtrack (false);
      else if (covered ())
	break;
      else
	lookahead ();
    }
  return !fail;
}

class SmodelsMain : public Smodels
{
public:
  SmodelsMain (SmodelsAux &);
  bool aset ();
  bool isminimal ();
  void reduce ();
  void unreduce ();
  void lookahead ();
  SmodelsAux &sm;
  AtomWithOther *no;
  unsigned long number_of_isminimal_calls;
};

SmodelsMain::SmodelsMain (SmodelsAux &sm0)
  : Smodels(), sm(sm0)
{
  no = 0;
  number_of_isminimal_calls = 0;
}

void
SmodelsMain::reduce ()
{
  for (long i = 0; i < stack.top; i++)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(stack.stack[i]);
      if (a->Bpos && a->other->hasvalue == false)
	{
	  Atom *b = a->other;
	  for (Follows *f = b->neg; f != b->endNeg; f++)
	    {
	      switch (f->r->type)
		{
		case BASICRULE:
		  if (static_cast<BasicRule *>(f->r)->head == no->other)
		    break;
		  f->r->inactivate (f);
		  static_cast<BasicRule *>(f->r)->lit++;
		  break;
		case CHOICERULE:
		  f->r->inactivate (f);
		  static_cast<ChoiceRule *>(f->r)->lit++;
		  break;
		default:
		  break;
		}
	    }
	}
    }
}

void
SmodelsMain::unreduce ()
{
  for (long i = 0; i < stack.top; i++)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(stack.stack[i]);
      if (a->Bpos && a->other->hasvalue == false)
	{
	  Atom *b = a->other;
	  for (Follows *f = b->neg; f != b->endNeg; f++)
	    {
	      switch (f->r->type)
		{
		case BASICRULE:
		  if (static_cast<BasicRule *>(f->r)->head == no->other)
		    break;
		  f->r->backtrackFromInactive (f);
		  static_cast<BasicRule *>(f->r)->lit--;
		  break;
		case CHOICERULE:
		  f->r->backtrackFromInactive (f);
		  static_cast<ChoiceRule *>(f->r)->lit--;
		  break;
		default:
		  break;
		}
	    }
	}
    }
}

void
SmodelsMain::lookahead ()
{
  if (lookahead_no_heuristic ())
    return;
  heuristic ();
  hi_is_positive = false; // Always try negative first
  choose ();
}

bool
SmodelsMain::isminimal ()
{
  long i;
  bool r = false;
  number_of_isminimal_calls++;
  reduce ();
  for (i = 0; i < program.number_of_atoms; i++)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(atom[i]);
      if (a->Bneg == true && a->other->Bneg == false)
	sm.setToBFalse (a->other);
    }
  sm.expand ();
  if (sm.conflict ())
    cout << "conflict" << endl;
  for (i = 0; i < stack.top; i++) // All true atoms must be in upper
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(stack.stack[i]);
      if (a->Bpos && !a->other->closure)
	{
	  cout << "success" << endl;
	  goto out;
	}
    }
  r = true;
  for (i = 0; i < program.number_of_atoms; i++)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(atom[i]);
      if (a->Bpos == false && a->other->Bneg == false)
	sm.setToBFalse (a->other);
    }
  sm.fail = false;
  if (!sm.model ())  // There doesn't have to be a model
    goto out;
  r = false;
  for (i = 0; i < stack.top; i++)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(stack.stack[i]);
      if (a->Bpos && !a->other->Bpos)
	goto out;
    }
  sm.backtrack (false);
  r = !sm.model ();  // There mustn't be any more models
 out:
  sm.unwind_to (sm.setup_top);
  unreduce ();
  return r;
}

bool
SmodelsMain::aset ()
{
  setup_with_lookahead ();
  if (conflict ())
    return 0;
  if (!covered ())
    improve (true);
  sm.setup ();

  while (!fail)
    {
      expand ();
      if (conflict ())
	backtrack (false);
      else if (covered ())
	{
	  if (isminimal ())
	    {
	      answer_number++;
	      cout << "Answer set: " << answer_number << endl;
	      printAnswer ();
	      program.set_optimum ();
	      if (max_models && answer_number >= max_models)
		return true;
	      backtrack (false);
	    }
	  else
	    {
	      do {
		backtrack (false);
		expand ();
	      } while (!fail && (conflict () || !isminimal ()));
	    }
	}
      else
      	lookahead ();
    }

  number_of_wrong_choices--;
  return false;
}

void print_time ()
{
  struct rusage rusage;
  getrusage (RUSAGE_SELF, &rusage);
  cout << "Duration: " << rusage.ru_utime.tv_sec << '.'
       << setw(3) << setfill('0') << rusage.ru_utime.tv_usec/1000
       << endl;
}

int main (int, char *[])
{
  SmodelsAux smodels_aux;
  SmodelsMain smodels(smodels_aux);
  ApiWithOther api (&smodels.program);
  Read reader (&smodels.program, &api);
  if (reader.read (cin))
    return 1;  // Error in input

  api.add_constraints ();
  // Make a copy that is used to test minimality
  ApiWithOther api_copy (&smodels_aux.program);
  api_copy.copy (&api);

  // We have to know which atoms are equal
  for (Node *n = smodels.program.atoms.head (); n; n = n->next)
    {
      AtomWithOther *a = static_cast<AtomWithOther *>(n->atom);
      AtomWithOther *b = static_cast<AtomWithOther *>(a->copy);
      a->other = b;
      b->other = a;
    }
  smodels.no = api.no;
  api_copy.done ();
  api.done ();
  smodels.init ();
  smodels.max_models = reader.models;
  smodels_aux.init ();

  smodels.shuffle ();
  if (smodels.aset ())
    cout << "True" << endl;
  else
    cout << "False" << endl;
  cout << "Number of isminimal calls: "
       << smodels.number_of_isminimal_calls << endl;

  print_time ();

  return 0;
}
