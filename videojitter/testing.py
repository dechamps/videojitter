import json
import sys


def prettify_json(path):
    with open(path) as file:
        contents = json.load(file)
    with open(path, "w") as file:
        json.dump(contents, file, indent=2)


def _reset_directory(path):
    path.mkdir(exist_ok=True)
    for child in path.iterdir():
        child.unlink()


def _write_directory_listing(path):
    file_names = [child.name for child in path.iterdir()]
    file_names.sort()
    with open(path / "file_list.txt", "w") as listing_file:
        for file_name in file_names:
            listing_file.write(f"{file_name}\n")


class Pipeline:
    def __init__(self, test_case):
        self._test_case = test_case

    def __enter__(self):
        _reset_directory(self.get_output_path())
        return self

    def __exit__(self, *kargs, **kwargs):
        _write_directory_listing(self.get_output_path())

    def get_output_path(self):
        return self._test_case.get_path() / "test_output"

    async def run_generate_spec(self, *args):
        await self._run_videojitter_module(
            "generate_spec",
            "--output-spec-file",
            self.get_spec_path(),
            *args,
        )
        prettify_json(self.get_spec_path())

    async def run_generate_video(self, *args):
        video_path = self.get_output_path() / "video.mp4"
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
            self.get_output_path() / "analyze_recording_debug_",
            *args,
        )

    async def run_generate_report(self, *args):
        json_report_chart_path = self.get_output_path() / "report.json"
        await self._run_videojitter_module(
            "generate_report",
            "--spec-file",
            self.get_spec_path(),
            "--frame-transitions-csv-file",
            self.get_frame_transitions_csv_path(),
            "--output-csv-file",
            self.get_output_path() / "report.csv",
            "--output-chart-file",
            json_report_chart_path,
            "--output-chart-file",
            self.get_output_path() / "report.html",
            *args,
        )
        prettify_json(json_report_chart_path)

    async def _run_videojitter_module(self, module_name, *args):
        with open(
            self.get_output_path() / f"{module_name}.stdout", "wb"
        ) as stdout, open(
            self.get_output_path() / f"{module_name}.stderr", "wb"
        ) as stderr:
            await self._test_case.run_subprocess(
                sys.executable,
                "-m",
                f"videojitter.{module_name}",
                *args,
                stdout=stdout,
                stderr=stderr,
            )

    def get_spec_path(self):
        return self.get_output_path() / "spec.json"

    def get_recording_path(self):
        return self.get_output_path() / "recording.wav"

    def get_frame_transitions_csv_path(self):
        return self.get_output_path() / "frame_transitions.csv"
