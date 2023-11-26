import asyncio

from videojitter_test import _pipeline


async def videojitter_test(test_case):
    with _pipeline.Pipeline(test_case) as pipeline:
        await pipeline.run_analyze_recording()
        await asyncio.gather(
            pipeline.run_generate_report(),
            pipeline.run_generate_report(
                "--chart-start-seconds",
                "34",
                "--chart-end-seconds",
                "38",
                prefix="zoomed_",
            ),
        )
