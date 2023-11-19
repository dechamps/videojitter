# videojitter developer documentation

![example workflow](https://github.com/dechamps/videojitter/actions/workflows/continuous-integration.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/videojitter.svg)](https://pypi.org/project/videojitter/)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

## Dev environment setup

1. Check out the repo.
2. Set up a [Python environment][].
3. Run the following from the root of the repo within the Python environment:

   ```shell
   pip3 install --editable .
   ```

4. `videojitter-*` commands should now be available in the environment.

## Testing

See [`videojitter_test/README`][].

[Python environment]: https://docs.python.org/3/tutorial/venv.html
[`videojitter_test/README`]: ../videojitter_test/README.md
