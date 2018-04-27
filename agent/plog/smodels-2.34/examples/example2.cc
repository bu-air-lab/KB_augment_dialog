// Run for example as follows: cat <file> | pparse | example2
#include <iostream>
#include "stable.h"

using namespace std;

int main ()
{
  Stable stable;
  int bad = stable.read (cin);  // Read from standard input
  if (bad)
    {
      cerr << "Error in input" << endl;
      return 1;
    }
  // Compute and display at most five stable models
  for (int i = 0; i < 5; i++)
    {
      // Every time model() is called a new stable model is computed
      if (stable.smodels.model () == 0) // If model() returns 0, then
	break;                          // there are no more models
      stable.smodels.printAnswer (); // Display the current model
    }
  return 0;
}
