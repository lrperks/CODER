#pragma once
#include "runtime/nfa.h"

// Shared with python/data/dsl.py (exported to Python as vedit.PROGRAM_PATTERN).
extern const char* const kProgramPattern;

// Matches one complete edit program, e.g. "replace(012,x + 1)\n".
const Nfa& program_nfa();
