import videojitter.testing


async def videojitter_test(test_case):
    await videojitter.testing.run_pipeline(test_case)
