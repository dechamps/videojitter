import sys


async def run_pipeline(test_case):
    spec_path = test_case.get_output_path("spec.json")
    await test_case.run_subprocess(
        "generate_spec",
        sys.executable,
        "-m",
        "videojitter.generate_spec",
        "--output-spec-file",
        spec_path,
    )
    recording_path = test_case.get_output_path("recording.wav")
    await test_case.run_subprocess(
        "generate_fake_recording",
        sys.executable,
        "-m",
        "videojitter.generate_fake_recording",
        "--spec-file",
        spec_path,
        "--output-recording-file",
        recording_path,
    )
    frame_transitions_csv_path = test_case.get_output_path("frame_transitions.csv")
    await test_case.run_subprocess(
        "analyze_recording",
        sys.executable,
        "-m",
        "videojitter.analyze_recording",
        "--spec-file",
        spec_path,
        "--recording-file",
        recording_path,
        "--output-frame-transitions-csv-file",
        frame_transitions_csv_path,
    )
    report_csv_path = test_case.get_output_path("report.csv")
    report_chart_path = test_case.get_output_path("report.json")
    await test_case.run_subprocess(
        "generate_report",
        sys.executable,
        "-m",
        "videojitter.generate_report",
        "--spec-file",
        spec_path,
        "--frame-transitions-csv-file",
        frame_transitions_csv_path,
        "--output-csv-file",
        report_csv_path,
        "--output-chart-file",
        report_chart_path,
    )
