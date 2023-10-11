import videojitter.testing


async def videojitter_test(test_case):
    pipeline = videojitter.testing.Pipeline(test_case)
    await pipeline.run_generate_spec()
    await pipeline.run_generate_fake_recording("--output-dtype", "int16")
    await pipeline.run_analyze_recording()
    await pipeline.run_generate_report()
