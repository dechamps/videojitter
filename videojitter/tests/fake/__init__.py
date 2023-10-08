import sys


async def videojitter_test(test_case):
    await test_case.run_subprocess(
        "generate_spec",
        sys.executable,
        "-m",
        "videojitter.generate_spec",
        "--output-spec-file",
        test_case.get_output_path("spec.json"),
    )
