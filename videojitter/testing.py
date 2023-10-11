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
        self._output_file_names = set()

    def __enter__(self):
        _reset_directory(self.get_output_path())
        return self

    def __exit__(self, *kargs, **kwargs):
        _write_directory_listing(self.get_output_path())

    def get_output_path(self):
        return self._test_case.get_path() / "test_output"

    def get_write_path(self, file_name):
        self._output_file_names.add(file_name)
        return self.get_output_path() / file_name

    def get_read_path(self, file_name, read_only_file_name=None):
        return (
            (self.get_output_path() / file_name)
            if file_name in self._output_file_names
            else (
                self._test_case.get_path()
                / (file_name if read_only_file_name is None else read_only_file_name)
            )
        )

    async def run_generate_spec(self, *args):
        json_path = self.get_write_path("spec.json")
        await self._run_videojitter_module(
            "generate_spec",
            "--output-spec-file",
            json_path,
            *args,
        )
        prettify_json(json_path)

    async def run_generate_video(self, *args):
        video_path = self.get_write_path("video.mp4")
        await self._run_videojitter_module(
            "generate_video",
            "--spec-file",
            self.get_read_path("spec.json"),
            "--output-file",
            video_path,
            *args,
        )
        assert video_path.stat().st_size > 0

    async def run_generate_fake_recording(self, *args):
        await self._run_videojitter_module(
            "generate_fake_recording",
            "--spec-file",
            self.get_read_path("spec.json"),
            "--output-recording-file",
            self.get_write_path("recording.wav"),
            *args,
        )

    async def run_analyze_recording(self, *args):
        await self._run_videojitter_module(
            "analyze_recording",
            "--spec-file",
            self.get_read_path("spec.json"),
            "--recording-file",
            self.get_read_path("recording.wav", "recording.flac"),
            "--output-frame-transitions-csv-file",
            self.get_write_path("frame_transitions.csv"),
            "--output-debug-files-prefix",
            self.get_output_path() / "analyze_recording_debug_",
            *args,
        )

    async def run_generate_report(self, *args):
        json_report_chart_path = self.get_write_path("report.json")
        await self._run_videojitter_module(
            "generate_report",
            "--spec-file",
            self.get_read_path("spec.json"),
            "--frame-transitions-csv-file",
            self.get_read_path("frame_transitions.csv"),
            "--output-csv-file",
            self.get_write_path("report.csv"),
            "--output-chart-file",
            json_report_chart_path,
            "--output-chart-file",
            self.get_write_path("report.html"),
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
