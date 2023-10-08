import asyncio
import importlib
import pkgutil
import subprocess
import sys


def main():
    class Driver:
        async def run_subprocess(self, name, *args):
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
            )
            await process.communicate()
            if process.returncode != 0:
                raise Exception(
                    f"Subprocess {name} terminated with error code {process.returncode}"
                )

    for test_module_info in pkgutil.iter_modules(["videojitter/tests"]):
        # TODO: parallelize
        asyncio.run(
            importlib.import_module(
                f"videojitter.tests.{test_module_info.name}"
            ).videojitter_test(Driver())
        )


if __name__ == "__main__":
    sys.exit(main())
