#include "runtime/grammar.h"

const char* const kProgramPattern = R"GRAM((noop\(\)|swap_op\([0-9]{3},[-+*/%<>=!]{1,2}\)|replace\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|change_return\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|add_guard\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|insert_before\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|delete_stmt\([0-9]{3}\)|swap_args\([0-9]{3},[0-9],[0-9]\))(;(noop\(\)|swap_op\([0-9]{3},[-+*/%<>=!]{1,2}\)|replace\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|change_return\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|add_guard\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|insert_before\([0-9]{3},[A-Za-z0-9 _.+\-*/%<>=!:,\[\]'"&|^~]+\)|delete_stmt\([0-9]{3}\)|swap_args\([0-9]{3},[0-9],[0-9]\))){0,2}\n)GRAM";

const Nfa& program_nfa() {
  static const Nfa nfa(kProgramPattern);
  return nfa;
}
