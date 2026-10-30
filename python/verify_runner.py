"""Child process for cpp/verify: runs one candidate and its test. Exit 0 means pass.

stdin: candidate source, a record separator (0x1e), then the test source.
argv[1]: name of the test function to call.
"""
import resource
import sys

MEM_LIMIT = 512 * 1024 * 1024  # address-space cap for the candidate

if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    src, test = sys.stdin.read().split("\x1e")
    ns = {}
    exec(compile(src, "<candidate>", "exec"), ns)
    exec(compile(test, "<test>", "exec"), ns)
    ns[sys.argv[1]]()
