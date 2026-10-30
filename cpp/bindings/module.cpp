#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "edits/edits.h"
#include "edits/prompt.h"
#include "runtime/decode.h"
#include "runtime/grammar.h"
#include "runtime/matmul.h"
#include "verify/solve.h"

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

  py::class_<Model>(m, "Model")
      .def(py::init<const std::string&>())
      .def_property_readonly("vocab", &Model::vocab);
  py::enum_<Mode>(m, "Mode").value("Edit", Mode::Edit).value("Free", Mode::Free);
  m.def("decode", [](const Model& model, const std::string& prompt, Mode mode, int max_new,
                     float temperature, uint64_t seed, bool stop_at_eos, bool avx2) {
    DecodeOpts o;
    o.mode = mode;
    o.max_new = max_new;
    o.temperature = temperature;
    o.seed = seed;
    o.stop_at_eos = stop_at_eos;
    o.avx2 = avx2;
    Decoded d;
    {
      py::gil_scoped_release release;  // only the C++ work; building the tuple needs the GIL
      d = decode(model, prompt, o);
    }
    return py::make_tuple(d.text, d.tokens, d.forced);
  }, py::arg("model"), py::arg("prompt"), py::arg("mode"), py::arg("max_new"),
     py::arg("temperature") = 0.0f, py::arg("seed") = 0, py::arg("stop_at_eos") = true,
     py::arg("avx2") = true);
  m.def("cpu_has_avx2", &cpu_has_avx2);
  m.def("run_tests", [](const std::vector<std::pair<std::string, std::string>>& jobs, int workers,
                        const std::string& python, const std::string& runner) {
    VerifyOpts v;
    v.python = python;
    v.runner = runner;
    std::vector<Job> js;
    for (auto& j : jobs) js.push_back({j.first, j.second});
    return run_pool(v, js, workers);
  }, py::call_guard<py::gil_scoped_release>());
  m.def("solve", [](const Model& model, const std::vector<std::tuple<std::string, std::string, std::string>>& rows,
                    Mode mode, int k, int max_new, int workers, uint64_t seed, bool avx2,
                    const std::string& python, const std::string& runner) {
    std::vector<Sample> samples;
    for (auto& r : rows) samples.push_back({std::get<0>(r), std::get<1>(r), std::get<2>(r)});
    SolveOpts o;
    o.mode = mode;
    o.k = k;
    o.max_new = max_new;
    o.workers = workers;
    o.seed = seed;
    o.avx2 = avx2;
    o.verify.python = python;
    o.verify.runner = runner;
    std::vector<py::tuple> res;
    for (const SampleStats& st : solve(model, samples, o)) {
      res.push_back(py::make_tuple(st.pass1, st.pass4, st.tokens, st.forced, st.verify_calls));
    }
    return res;
  }, py::call_guard<py::gil_scoped_release>());
}
