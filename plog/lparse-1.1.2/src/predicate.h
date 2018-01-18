// predicate.h -- declarations for predicate data type for lparse
// Copyright (C) 1999-2007 Tommi Syrjänen <Tommi.Syrjanen@hut.fi>
//  
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//  
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//  
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//  

#ifndef PREDICATE_H
#define PREDICATE_H

#include "config.h"
#ifndef GLOBAL_H
#include "global.h"
#endif
#ifndef LIST_H
#include "list.h"
#endif
#ifndef LITERAL_H
#include "literal.h"
#endif
#ifndef RULE_H
#include "rule.h"
#endif
#ifndef INSTANCE_H
#include "instance.h"
#endif
#ifndef GRAPH_H
#include "graph.h"
#endif

#include "smodels.h"

class Predicate
{
public:
  Predicate();
  ~Predicate();

  void AddRule(Rule *, int cmp = 1);
  void AddRange(Range *);
  void AddSpecialRule(Rule *, int cmp = 1);
  void AddWeight(Weight *nw, Literal *lt, int global);
  void AddInstance(Instance *inst);
  void CreateIndex(int index);
  void Emit();
  void PrintRules();
  
  int Arity() { return arity; }
  int HasSpecial() { return special; }
  void SetArity(int ar);
  void SetPred(long p) { pred = p; }
  long Pred() { return pred; }
  const char *Name() { return name; }
  void SetName(const char *nm) { name = nm; }
  DomainType Status() { return status; }
  void SetStatus(DomainType r);
  long Size();
  long ExtensionSize();
  void CalculateDomain();
  
  RestrictType CheckRestrict(int print);
  static RestrictType CheckAllRestricts();
  static void EmitAll();
  static void PrintAllRules();
  static long MakeComplement(long, const char *, int);

  static long MakeNegated(long, const char *, int);
  
  static long DefinePredicate(const char *, int, long);
  static long DefineSystemPredicate(DomainType tp = DM_INTERNAL);
  static long DefineRangePredicate();

  static long DefinePriorityPredicate();

  static long ConstructDummy();

  static void DefineAllComplements();
  static void CreateAllComplementRules();

  static void DefineImplicitDomain(LiteralList *lst);
  static void CheckImplicitDomains();
  
  void CreateComplementRules();

  void AddSCCFacts();
  
  Instance GetArgumentInstanceFromString(const char *st, int n);
  // initialize a complementary predicate to allow regular model
  // semantics to work 
  static long DefineComplement(long pred);
  static long DefineNormalRuleComplement(long pred);

  static void DefineNegated(long pred);

  static Literal *DefineFalseLiteral();
  static long DefineInconsistentPredicate();
  
  static void EmitComplements();

  static void EmitNegations();

  int DomainPredicate() { return status == DM_DOMAIN ||
			    status == DM_INTERNAL; }
  
  // tries to find predicates without any rules for them 
  static void CheckUnsatisfiable();
  static void PrintDebugData();
  
  int NeedsToBePrinted();

  int Internal(); // is the predicate internal or not
  
  static long decode_atom(const char *st);
  static long decode_atom(const char *st, Instance *it);
  static void DefineImplicitDomains();

  void DefineImplicitDomain();


  static int HasSpecialRules(StronglyConnectedComponent *scc);
  static void CalculateSCCDomain(StronglyConnectedComponent *scc);
  static void SetDomainFalse(StronglyConnectedComponent *scc);

  static void PrintDomainPredicates();
  static void PrintDomainRules();

  static void DefineSignature(long *);
  
  Index **indices;
  InstanceSet *atoms;
  InstanceSet *emitted;
  unsigned long existing_indexes;
  long complement; // the complementary predicate for regular models
  long normalized_complement; // the complementary predicate for regular models
  long negation; // the negated predicate for classical negation
  int arity;
  bool follows;
  bool special;
  bool hidden;
  bool shown;
  bool transitive_domain; // is it a transitive domain predicate.
  bool external_atom;
  bool depends_on_external_atoms;
  bool has_rule; // true if predicate occurs as a head
  bool is_warned; // have we warned about mismatches of this
  		     // predicate
  bool is_internal;
  bool is_priority_predicate;
  bool is_priority_meta_predicate;
  bool is_priority_tcl;
  bool is_only_symbolic_function;
  
  WeightNode *predicate_weight_list;
  WeightNode *weight_list;

  long first_line; // line of user code where the predicate occurs
		   //first

  int num_components;
  int component;
  long *signature;
  
  static lp_smodels *transitive_program;
private:
  RuleList rules;
  RuleList special_rules;
  RangeList ranges;
  
  InstanceIterator *it;
  const char *name;
  long pred;

  DomainType status;


};
#endif
