#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "edits/edits.h"
#include "edits/prompt.h"
#include "runtime/grammar.h"

namespace py = pybind11;

PYBIND11_MODULE(vedit, m) {
  m.doc() = "Verified edit-program runtime: tree-sitter edits, DSL grammar, prompts.";
  m.attr("PROGRAM_PATTERN") = kProgramPattern;
  m.def("render_prompt", &render_prompt, py::arg("test"), py::arg("src"), py::arg("annotated"));
  m.def("valid_program", [](const std::string& text) { return program_nfa().matches(text); });
  m.def("apply_program", [](const std::string& src, const std::string& program) -> py::object {
    auto prog = parse_program(program);
    if (!prog) return py::none();
    auto out = apply_program(src, *prog);
    return out ? py::cast(*out) : py::none();
  });
  m.def("grammar_allowed", [](const std::string& prefix) {
    const Nfa& nfa = program_nfa();
    Nfa::States s = nfa.start();
    for (char c : prefix) s = nfa.step(s, c);
    std::vector<int> out;
    auto bits = nfa.allowed(s);
    for (int c = 0; c < 128; ++c) {
      if (bits.test(c)) out.push_back(c);
    }
    return out;
  });
}
