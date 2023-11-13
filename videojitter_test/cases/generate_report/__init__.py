from videojitter_test import _pipeline


async def videojitter_test(test_case):
    with _pipeline.Pipeline(test_case) as pipeline:
        await pipeline.run_generate_report(
            "--keep-first-transition", "--keep-last-transition"
        )
