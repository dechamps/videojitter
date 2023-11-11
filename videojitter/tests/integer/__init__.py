import videojitter._testing


async def videojitter_test(test_case):
    with videojitter._testing.Pipeline(test_case) as pipeline:
        await pipeline.run_generate_spec()
        await pipeline.run_generate_fake_recording("--output-sample-type", "PCM_16")
        await pipeline.run_analyze_recording()
        await pipeline.run_generate_report()
