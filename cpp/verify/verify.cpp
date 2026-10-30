#include "verify/verify.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <thread>

extern char** environ;

namespace {

using Clock = std::chrono::steady_clock;

bool write_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t w = write(fd, s.data() + off, s.size() - off);
    if (w <= 0) return false;
    off += static_cast<size_t>(w);
  }
  return true;
}

// Reads the worker's one-byte reply. Returns '0' or '1', or -1 on deadline or EOF.
int read_reply(int fd, Clock::time_point deadline) {
  for (;;) {
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (left <= 0) return -1;
    pollfd p{fd, POLLIN, 0};
    int r = poll(&p, 1, static_cast<int>(left));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    char c;
    return read(fd, &c, 1) == 1 ? c : -1;
  }
}

}  // namespace

Worker::Worker(VerifyOpts opts) : opts_(std::move(opts)) {}

Worker::~Worker() { kill_worker(); }

bool Worker::spawn() {
  signal(SIGPIPE, SIG_IGN);  // a dead worker must show up as a failed write, not a signal
  int in[2], out[2];
  // CLOEXEC: a worker spawned by another thread must not inherit our ends, or EOF never arrives.
  if (pipe2(in, O_CLOEXEC) != 0) return false;
  if (pipe2(out, O_CLOEXEC) != 0) {
    close(in[0]);
    close(in[1]);
    return false;
  }
  posix_spawn_file_actions_t fa;
  posix_spawnattr_t attr;
  posix_spawn_file_actions_init(&fa);
  posix_spawnattr_init(&attr);
  posix_spawn_file_actions_adddup2(&fa, in[0], 0);
  posix_spawn_file_actions_adddup2(&fa, out[1], 1);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attr, 0);  // own process group, so a kill also takes its test children
  char* argv[] = {const_cast<char*>(opts_.python.c_str()), const_cast<char*>("-I"),
                  const_cast<char*>(opts_.runner.c_str()), nullptr};
  pid_t pid;
  int rc = posix_spawnp(&pid, opts_.python.c_str(), &fa, &attr, argv, environ);  // p: finds "python3" on PATH
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&attr);
  close(in[0]);
  close(out[1]);
  if (rc != 0) {
    close(in[1]);
    close(out[0]);
    return false;
  }
  pid_ = pid;
  to_ = in[1];
  from_ = out[0];
  return true;
}

void Worker::kill_worker() {
  if (pid_ < 0) return;
  kill(-pid_, SIGKILL);  // the worker and any test child it forked
  waitpid(pid_, nullptr, 0);
  close(to_);
  close(from_);
  pid_ = -1;
}

bool Worker::run(const std::string& src, const std::string& test) {
  if (pid_ < 0 && !spawn()) return false;
  Clock::time_point deadline =
      Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(opts_.timeout_s));
  std::string req = opts_.test_name + "\n" + std::to_string(src.size()) + " " +
                    std::to_string(test.size()) + "\n" + src + test;
  if (!write_all(to_, req)) {
    kill_worker();
    return false;
  }
  int reply = read_reply(from_, deadline);
  if (reply != '0' && reply != '1') {  // timeout or dead worker
    kill_worker();
    return false;
  }
  return reply == '1';
}

std::vector<bool> run_pool(const VerifyOpts& opts, const std::vector<Job>& jobs, int workers) {
  std::vector<char> pass(jobs.size(), 0);  // not vector<bool>: workers write neighbouring elements
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&] {
      Worker worker(opts);
      for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
        pass[i] = worker.run(jobs[i].src, jobs[i].test);
      }
    });
  }
  for (auto& t : pool) t.join();
  return std::vector<bool>(pass.begin(), pass.end());
}
