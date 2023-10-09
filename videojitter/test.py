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
        help="Run the specified test case. Can be specified multiple times. If not set, runs all test cases.",
        action="append",
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--parallelism",
        help="How many test cases to run in parallel.",
        type=int,
        default=len(os.sched_getaffinity(0)),
    )
    return argument_parser.parse_args()


def _reset_directory(path):
    path.mkdir(exist_ok=True)
    for child in path.iterdir():
        child.unlink()


class _TestCase:
    def __init__(self, root_directory, name):
        self.name = name
        self.module = importlib.import_module(f"videojitter.tests.{name}")
        self.output_dir = root_directory / name / "test_output"

    async def run(self):
        try:
            _reset_directory(self.output_dir)
            await self.module.videojitter_test(self)
        except Exception as exception:
            raise Exception(f"Failed to run test: {self.name}") from exception

    def get_output_path(self, file_name):
        return self.output_dir / file_name

    async def run_subprocess(self, name, *args):
        args = [str(arg) for arg in args]
        print(f"{self.name}: running {name}: {args}")
        with open(self.get_output_path(f"{name}.stdout"), "wb") as stdout, open(
            self.get_output_path(f"{name}.stderr"), "wb"
        ) as stderr:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            await process.communicate()
            if process.returncode != 0:
                raise Exception(
                    f"Subprocess {name} terminated with error code {process.returncode}"
                )


async def _run_tests():
    args = _parse_arguments()
    test_cases = getattr(args, "test_case", None)
    tests_directory = pathlib.Path("videojitter") / "tests"
    throttle = asyncio.Semaphore(args.parallelism)

    async def run(test_case):
        async with throttle:
            await test_case.run()

    async with asyncio.TaskGroup() as task_group:
        for test_module_name in (
            [
                module_info.name
                for module_info in pkgutil.iter_modules([tests_directory])
            ]
            if test_cases is None
            else test_cases
        ):
            task_group.create_task(run(_TestCase(tests_directory, test_module_name)))


def main():
    asyncio.run(_run_tests())


if __name__ == "__main__":
    sys.exit(main())
