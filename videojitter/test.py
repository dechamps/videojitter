import asyncio
import importlib
import pathlib
import pkgutil
import subprocess
import sys


def _reset_directory(path):
    path.mkdir(exist_ok=True)
    for child in path.iterdir():
        child.unlink()


class _TestCase:
    def __init__(self, name):
        self.name = name
        self.module = importlib.import_module(f"videojitter.tests.{name}")
        self.output_dir = pathlib.Path("videojitter") / "tests" / name / "test_output"

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
    tests_directory = pathlib.Path("videojitter") / "tests"
    async with asyncio.TaskGroup() as task_group:
        for test_module_info in pkgutil.iter_modules([tests_directory]):
            task_group.create_task(_TestCase(test_module_info.name).run())


def main():
    asyncio.run(_run_tests())


if __name__ == "__main__":
    sys.exit(main())
