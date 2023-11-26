# videojitter testing

## How to run the test suite

Follow the [developer setup instructions][], then install the test-specific
dependencies into the Python environment:

```shell
pip3 install -r test-requirements.txt
```

You can then run the test suite using the following command at the root
directory:

```shell
python3 -m videojitter_test
```

The test suite is successful if:

- It completes without errors; and
- It doesn't cause any git diffs in `test_output` directories.

## How the test suite works

Every Python module in the [`cases` directory][] is called a _test case_, and
comes with a special entry point for the test driver to call.

The above command runs the test suite driver, which runs all the test cases in
parallel. (To customize this behaviour, run the command with `--help`).

A typical test case runs a series of videojitter commands (the _pipeline_) as
subprocesses, feeding the output of one command to the next command. For more
details on what individual test cases do, see [`cases/README`][].

The various outputs of each videojitter command (stdout, stderr, and output
files) are written to the `test_outputs` subdirectory of each test case.

Besides checking that the commands run successfully, the test suite also acts as
a _change detector_ test (a.k.a "golden testing"). This is done by checking in
some of the test output files (the _goldens_) in version control, such that any
changes to the output triggers a git diff. This alerts the user of a potentially
unexpected change to the outputs and provides a convenient mechanism for
inspecting the diffs through the usual git tools.

## What if I cause a git diff on the goldens (`test_outputs`)?

Obviously, some diffs are normal and expected if the point of your change is to
alter videojitter outputs. In that case, the correct course of action is to
simply include the resulting test outputs in your commit, thus making them the
new goldens.

If the diffs show an unacceptable regression, fix your code to stop producing
broken output then run the test suite again.

Some changes, especially changes to the analyzer math, can lead to benign diffs
such as very small changes to the last decimal of some numbers due to very
slight differences in numerical precision. It's usually fine to just shrug these
off and update the goldens with the new numbers.

[developer setup instructions]: ../src/README.md
[`cases` directory]: cases/
[`cases/README`]: cases/README.md
