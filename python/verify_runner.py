"""Persistent worker for cpp/verify. Each test runs in a fork of this process under a memory cap.

Request (stdin): "<test name>\n<len(src)> <len(test)>\n", then the candidate source and the test source.
Reply (stdout): one byte, "1" if the test passed and "0" otherwise. The forked child's stdout and
stderr go to /dev/null, so nothing else reaches the reply stream.
"""
import os
import resource
import sys

MEM_LIMIT = 512 * 1024 * 1024  # address-space cap per test


def run_child(src, test, name):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        ns = {}
        exec(compile(src, "<candidate>", "exec"), ns)
        exec(compile(test, "<test>", "exec"), ns)
        ns[name]()
    except BaseException:
        os._exit(1)
    os._exit(0)


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while head := inp.readline():
        name = head.decode().strip()
        n_src, n_test = map(int, inp.readline().split())
        src = inp.read(n_src).decode()
        test = inp.read(n_test).decode()
        pid = os.fork()
        if pid == 0:
            run_child(src, test, name)
        _, status = os.waitpid(pid, 0)
        out.write(b"1" if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else b"0")
        out.flush()


if __name__ == "__main__":
    main()
