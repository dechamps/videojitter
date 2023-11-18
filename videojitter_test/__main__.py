import argparse
import asyncio
import importlib
import os
import pathlib
import pkgutil
import subprocess
import sys


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Runs the videojitter test suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--test-case",
        help=(
            "Run the specified test case. Can be specified multiple times. If not set,"
            " runs all test cases."
        ),
        action="append",
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--parallelism",
        help="How many test cases to run in parallel.",
        type=int,
        default=(
            len(os.sched_getaffinity(0))
            if "sched_getaffinity" in dir(os)
            else os.cpu_count()
        ),
    )
    return argument_parser.parse_args()


class _TestCase:
    def __init__(self, root_directory, name):
        self._name = name
        self._module = importlib.import_module(f"videojitter_test.cases.{name}")
        self._path = root_directory / name

    async def run(self):
        try:
            await self._module.videojitter_test(self)
        except Exception as exception:
            raise RuntimeError(f"Failed to run test: {self._name}") from exception

    def get_path(self):
        return self._path

    async def run_subprocess(self, *args, env, stdout, stderr):
        args = [str(arg) for arg in args]
        print(f"{self._name}: {args}")
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=subprocess.DEVNULL,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"Subprocess terminated with error code {process.returncode}"
            )


async def _run_tests():
    args = _parse_arguments()
    test_cases = getattr(args, "test_case", None)
    tests_directory = pathlib.Path("videojitter_test") / "cases"
    throttle = asyncio.Semaphore(args.parallelism)

    async def run(test_case):
        async with throttle:
            await test_case.run()

    # Note: asyncio.TaskGroup would normally be preferred, but it would force us to
    # require Python 3.11 which is still a bit fresh at the time of writing.
    await asyncio.gather(*[
        run(_TestCase(tests_directory, test_module_name))
        for test_module_name in (
            [
                module_info.name
                # Note str() is necessary due to https://bugs.python.org/issue44061
                for module_info in pkgutil.iter_modules([str(tests_directory)])
            ]
            if test_cases is None
            else test_cases
        )
    ])


def main():
    asyncio.run(_run_tests())


if __name__ == "__main__":
    sys.exit(main())
