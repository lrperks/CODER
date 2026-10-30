#include <gtest/gtest.h>

#include <chrono>

#include "verify/verify.h"

namespace {

const char* kAdd = "def f(a, b):\n    return a + b\n";
const char* kTest = "def test_f():\n    assert f(1, 2) == 3\n";

VerifyOpts opts() {
  VerifyOpts o;
  o.runner = VEDIT_RUNNER;
  return o;
}

bool run_once(const std::string& src, const std::string& test) { return Worker(opts()).run(src, test); }

}  // namespace

TEST(Verify, PassingCandidate) { EXPECT_TRUE(run_once(kAdd, kTest)); }

TEST(Verify, FailingAssertion) { EXPECT_FALSE(run_once("def f(a, b):\n    return a - b\n", kTest)); }

TEST(Verify, SyntaxErrorCountsAsFailure) { EXPECT_FALSE(run_once("def f(a, b) return a + b\n", kTest)); }

TEST(Verify, CandidateOutputDoesNotFakeAReply) {
  EXPECT_FALSE(run_once("def f(a, b):\n    print('1' * 100)\n    return a - b\n", kTest));
}

TEST(Verify, TimeoutKillsRunawayCandidate) {
  auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(run_once("def f(a, b):\n    while True: pass\n", kTest));
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_LT(s, 3.5);  // 2 s limit plus scheduling slack
}

TEST(Verify, MemoryCapFailsAllocation) {
  EXPECT_FALSE(run_once("def f(a, b):\n    x = bytearray(1 << 30)\n    return a + b\n", kTest));
}

TEST(Verify, WorkerPersistsAcrossTestsAndRespawnsAfterTimeout) {
  Worker w(opts());
  for (int i = 0; i < 20; ++i) EXPECT_TRUE(w.run(kAdd, kTest)) << i;
  EXPECT_FALSE(w.run("def f(a, b):\n    while True: pass\n", kTest));
  EXPECT_TRUE(w.run(kAdd, kTest));
}

TEST(Verify, PoolPreservesOrderAtAnyWorkerCount) {
  std::vector<Job> jobs;
  for (int i = 0; i < 12; ++i) {
    jobs.push_back({i % 3 == 0 ? "def f(a, b):\n    return a - b\n" : kAdd, kTest});
  }
  std::vector<bool> expect;
  for (int i = 0; i < 12; ++i) expect.push_back(i % 3 != 0);
  for (int w : {1, 4, 8}) EXPECT_EQ(run_pool(opts(), jobs, w), expect) << "workers=" << w;
}
