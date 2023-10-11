import json
import sys


def prettify_json(path):
    with open(path) as file:
        contents = json.load(file)
    with open(path, "w") as file:
        json.dump(contents, file, indent=2)


class Pipeline:
    def __init__(self, test_case):
        self._test_case = test_case

    async def run_generate_spec(self, *args):
        await self._run_videojitter_module(
            "generate_spec",
            "--output-spec-file",
            self.get_spec_path(),
            *args,
        )
        prettify_json(self.get_spec_path())

    async def run_generate_video(self, *args):
        video_path = self._test_case.get_output_path("video.mp4")
        await self._run_videojitter_module(
            "generate_video",
            "--spec-file",
            self.get_spec_path(),
            "--output-file",
            video_path,
            *args,
        )
        assert video_path.stat().st_size > 0

    async def run_generate_fake_recording(self, *args):
        await self._run_videojitter_module(
            "generate_fake_recording",
            "--spec-file",
            self.get_spec_path(),
            "--output-recording-file",
            self.get_recording_path(),
            *args,
        )
        prettify_json(self.get_spec_path())

    async def run_analyze_recording(self, *args):
        await self._run_videojitter_module(
            "analyze_recording",
            "--spec-file",
            self.get_spec_path(),
            "--recording-file",
            self.get_recording_path(),
            "--output-frame-transitions-csv-file",
            self.get_frame_transitions_csv_path(),
            "--output-debug-files-prefix",
            self._test_case.get_output_path("analyze_recording_debug_"),
            *args,
        )

    async def run_generate_report(self, *args):
        json_report_chart_path = self._test_case.get_output_path("report.json")
        await self._run_videojitter_module(
            "generate_report",
            "--spec-file",
            self.get_spec_path(),
            "--frame-transitions-csv-file",
            self.get_frame_transitions_csv_path(),
            "--output-csv-file",
            self._test_case.get_output_path("report.csv"),
            "--output-chart-file",
            json_report_chart_path,
            "--output-chart-file",
            self._test_case.get_output_path("report.html"),
            *args,
        )
        prettify_json(json_report_chart_path)

    async def _run_videojitter_module(self, module_name, *args):
        await self._test_case.run_subprocess(
            module_name, sys.executable, "-m", f"videojitter.{module_name}", *args
        )

    def get_spec_path(self):
        return self._test_case.get_output_path("spec.json")

    def get_recording_path(self):
        return self._test_case.get_output_path("recording.wav")

    def get_frame_transitions_csv_path(self):
        return self._test_case.get_output_path("frame_transitions.csv")
