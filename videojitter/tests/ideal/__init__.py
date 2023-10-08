import videojitter.testing


async def videojitter_test(test_case):
    await videojitter.testing.run_pipeline(
        test_case,
        generate_fake_recording_args=[
            "--clock-skew",
            1,
            "--pattern-count",
            0,
            "--white-duration-overshoot",
            0,
            "--even-duration-overshoot",
            0,
            "--dc-offset",
            0,
            "--gaussian-filter-stddev-seconds",
            0,
            "--high-pass-filter-hz",
            0,
            "--noise-rms-per-hz",
            0,
        ],
    )
