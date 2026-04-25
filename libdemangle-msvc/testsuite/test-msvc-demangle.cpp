#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sstream>

#include <fstream>
#include <iostream>

#include "demangle.h"

int
main (int argc, char *argv[])
{
  int tests = 0;
  int passes = 0;
  int failures = 0;
  int lineno = 0;
  std::string mangled, expected;

  // Read from stdin in pairs: mangled name [options], then expected demangled output
  while (std::getline (std::cin, mangled))
    {
      lineno++;
      
      // Skip empty lines and comments
      if (mangled.empty() || mangled[0] == '#')
        continue;

      // Skip lines that don't look like mangled names
      if (mangled[0] != '?' && mangled[0] != '_')
        continue;

      // Parse optional flags from mangled line
      int flags = DMGL_TYPES | DMGL_PARAMS;
      size_t first_space = mangled.find(' ');
      
      if (first_space != std::string::npos)
        {
          std::string symbol = mangled.substr(0, first_space);
          std::string options_str = mangled.substr(first_space + 1);
          
          // Parse options
          std::istringstream iss(options_str);
          std::string option;
          while (iss >> option)
            {
              if (option == "NO_PARAMS") 
                flags &= ~DMGL_PARAMS;
              if (option == "MSVC_FULL")
                flags |= DMGL_MSVC;
            }
          mangled = symbol;
        }

      // Read expected demangled output
      if (!std::getline (std::cin, expected))
        {
          fprintf (stderr, "Error: line %d - missing expected output for mangled name\n", lineno);
          failures++;
          continue;
        }
      lineno++;

      tests++;

      char *demangled = msvc_demangle (mangled.c_str(), flags);
      
      if (demangled)
        {
          if (expected == demangled)
            {
              printf ("PASS: %s\n", mangled.c_str());
              passes++;
            }
          else
            {
              printf ("FAIL: %s\n", mangled.c_str());
              printf ("  Expected: %s\n", expected.c_str());
              printf ("  Got:      %s\n", demangled);
              failures++;
            }
          free (demangled);
        }
      else
        {
          printf ("FAIL: %s (demangling returned NULL)\n", mangled.c_str());
          printf ("  Expected: %s\n", expected.c_str());
          failures++;
        }
    }
  
  // Summary
  printf ("\n");
  printf ("===================\n");
  printf ("Test Summary\n");
  printf ("===================\n");
  printf ("Total tests:  %d\n", tests);
  printf ("PASS:         %d\n", passes);
  printf ("FAIL:         %d\n", failures);
  
  if (failures == 0)
    printf ("\nAll tests passed!\n");
  else
    printf ("\n%d test%s failed.\n", failures, failures == 1 ? "" : "s");
  
  return (failures > 0) ? 1 : 0;
}
