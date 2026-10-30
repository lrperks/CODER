#pragma once
#include <string>

// Builds the policy input: the test, then the source. With annotated=true every
// named node and operator token is prefixed with its "{id}" so the policy can
// refer to nodes by id.
std::string render_prompt(const std::string& test, const std::string& src, bool annotated);
