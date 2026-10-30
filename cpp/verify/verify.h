#pragma once
#include <string>
#include <vector>

struct VerifyOpts {
  std::string python = "python3";        // interpreter for the child process
  std::string runner;                    // path to python/verify_runner.py
  std::string test_name = "test_f";
  double timeout_s = 2.0;
};

struct Job {
  std::string src, test;
};

// Runs one candidate against its test in a child process. True if the test passes
// within the timeout. Timeouts, crashes and syntax errors all count as failures.
bool run_test(const VerifyOpts& opts, const std::string& src, const std::string& test);

// Runs jobs on `workers` concurrent child processes; returns one pass flag per job.
std::vector<bool> run_pool(const VerifyOpts& opts, const std::vector<Job>& jobs, int workers);
