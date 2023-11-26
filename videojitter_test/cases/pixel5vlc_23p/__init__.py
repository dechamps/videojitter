from videojitter_test import _pipeline


async def videojitter_test(test_case):
    with _pipeline.Pipeline(test_case) as pipeline:
        await pipeline.run_analyze_recording()
        # TODO: edge direction compensation fails badly on this example, leading to
        # nonsensical results. This is because the test signal is shaped by a 3:2
        # 23p@60Hz signal that happens to "flip" every 10 seconds. The intentionally
        # delayed transition, which occurs at 30 seconds, cancels out one of these
        # flips. This leads to white frames being displayed for far longer than black
        # frames on average, messing up the falling/rising edge lag estimate.
        await pipeline.run_generate_report()
