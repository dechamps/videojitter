from videojitter import _testing


async def videojitter_test(test_case):
    with _testing.Pipeline(test_case) as pipeline:
        await pipeline.run_generate_spec()
        await pipeline.run_generate_fake_recording("--gain", 2)
        await pipeline.run_analyze_recording()
        await pipeline.run_generate_report()
