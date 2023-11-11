from videojitter import _testing


async def videojitter_test(test_case):
    with _testing.Pipeline(test_case) as pipeline:
        # Encoding video takes time. Use small durations so that the test
        # finishes quickly.
        await pipeline.run_generate_spec("--duration-seconds", 2)
        await pipeline.run_generate_video("--begin-padding", 1, "--end-padding", 1)
