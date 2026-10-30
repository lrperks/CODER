#pragma once
#include <string>
#include <vector>

struct VerifyOpts {
  std::string python = "python3";        // interpreter for the worker process
  std::string runner;                    // path to python/verify_runner.py
  std::string test_name = "test_f";
  double timeout_s = 2.0;                // per test, including the fork
};

struct Job {
  std::string src, test;
};

// A persistent Python worker process. Each test runs in a fork of the worker, so no interpreter
// startup is paid per test. A test that times out gets its whole process group killed, and the
// worker is respawned on next use. Timeouts, crashes and syntax errors all count as failures.
class Worker {
 public:
  explicit Worker(VerifyOpts opts);
  ~Worker();
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  bool run(const std::string& src, const std::string& test);

 private:
  bool spawn();
  void kill_worker();

  VerifyOpts opts_;
  int pid_ = -1;
  int to_ = -1;    // request pipe, our write end
  int from_ = -1;  // reply pipe, our read end
};

// Runs jobs on `workers` threads, each with its own Worker; returns one pass flag per job.
std::vector<bool> run_pool(const VerifyOpts& opts, const std::vector<Job>& jobs, int workers);
