# si-prefix

This directory contains a copy of the main code of [si-prefix v1.2.2][] by
Christian Fobel (with a couple of minor compatibility changes).

The reason why videojitter does not simply pull si-prefix as a pip dependency is
because the si-prefix build [fails on Python 3.12][].

[si-prefix v1.2.2]: https://github.com/cfobel/si-prefix/tree/v1.2.2
[fails on Python 3.12]: https://github.com/cfobel/si-prefix/issues/11
