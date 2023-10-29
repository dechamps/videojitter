import videojitter.testing


async def videojitter_test(test_case):
    with videojitter.testing.Pipeline(test_case) as pipeline:
        await pipeline.run_generate_report(
            "--keep-first-transition",
            "--keep-last-transition",
            "--edge-direction-compensation",
        )
