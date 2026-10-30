#include "verify/verify.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>

extern char** environ;

namespace {

constexpr char kSep = '\x1e';

}  // namespace

bool run_test(const VerifyOpts& opts, const std::string& src, const std::string& test) {
  signal(SIGPIPE, SIG_IGN);  // the child may exit before reading its input
  int in[2];
  // CLOEXEC: a child spawned by another thread must not inherit our write end, or it never sees EOF.
  if (pipe2(in, O_CLOEXEC) != 0) return false;
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, in[0], 0);
  posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addclose(&fa, in[1]);
  char* argv[] = {const_cast<char*>(opts.python.c_str()), const_cast<char*>("-I"),
                  const_cast<char*>(opts.runner.c_str()), const_cast<char*>(opts.test_name.c_str()),
                  nullptr};
  pid_t pid;
  int rc = posix_spawnp(&pid, opts.python.c_str(), &fa, nullptr, argv, environ);  // p: finds "python3" on PATH
  posix_spawn_file_actions_destroy(&fa);
  close(in[0]);
  if (rc != 0) {
    close(in[1]);
    return false;
  }
  std::string payload = src + kSep + test;
  size_t off = 0;
  while (off < payload.size()) {
    ssize_t w = write(in[1], payload.data() + off, payload.size() - off);
    if (w <= 0) break;
    off += static_cast<size_t>(w);
  }
  close(in[1]);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(opts.timeout_s);
  int status = 0;
  for (;;) {
    pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::vector<bool> run_pool(const VerifyOpts& opts, const std::vector<Job>& jobs, int workers) {
  std::vector<char> pass(jobs.size(), 0);  // not vector<bool>: workers write neighbouring elements
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
        pass[i] = run_test(opts, jobs[i].src, jobs[i].test);
      }
    });
  }
  for (auto& t : pool) t.join();
  return std::vector<bool>(pass.begin(), pass.end());
}
