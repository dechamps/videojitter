import videojitter.testing


async def videojitter_test(test_case):
    pipeline = videojitter.testing.Pipeline(test_case)
    # Encoding video takes time. Use small durations so that the test finishes
    # quickly.
    await pipeline.run_generate_spec("--duration-seconds", 2)
    await pipeline.run_generate_video("--begin-padding", 1, "--end-padding", 1)
