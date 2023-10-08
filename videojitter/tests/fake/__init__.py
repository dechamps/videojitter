import sys


async def videojitter_test(driver):
    await driver.run_subprocess(
        "generate_spec", sys.executable, "-m", "videojitter.generate_spec"
    )
