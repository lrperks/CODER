#pragma once
#include <optional>
#include <string>
#include <vector>

#include "edits/program.h"

// Applies a program to src. Node ids refer to the parse of src, so all edits are
// computed on that parse and must not overlap. Returns nullopt if a primitive
// does not apply or the edited text has ERROR/MISSING nodes.
std::optional<std::string> apply_program(const std::string& src, const std::vector<Prim>& prog);
