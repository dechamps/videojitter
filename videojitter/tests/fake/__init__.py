import sys


async def videojitter_test(test_case):
    await test_case.run_subprocess(
        "generate_spec", sys.executable, "-m", "videojitter.generate_spec"
    )
