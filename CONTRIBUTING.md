# How to Contribute

We welcome community contributions to the oneAPI-samples project. You can:

- Submit your changes directly as a [pull request](#pull-requests)
- Log a bug or feature request with an [issue](#issues)

## Pull requests

We accept contributions as pull requests on GitHub. To contribute your changes
directly to the repository, please do the following:

* Follow the [Git Steps for contribution](https://github.com/oneapi-src/oneAPI-samples/wiki/Git-Steps-for-Contribution).
* Make sure your PR has a clear purpose, and does one thing only, and nothing more. This will enable us review your PR more quickly.
* Make sure each commit in your PR is a small, atomic change representing one step in development.
* Squash intermediate steps within PR for bug fixes, style cleanups, reversions, etc., so they do not appear in merged PR history.
* Explain anything non-obvious from the code in comments, commit messages, or the PR description, as appropriate.

## Issues

GitHub Issues tracks sample development issues, bugs, and feature requests.
For usage, installation, or other requests for help, please use the [Intel® oneAPI Forums](https://software.intel.com/en-us/forums/intel-oneapi-forums) instead.

When reporting a bug, please provide the following information, where applicable:

* What are the steps to reproduce the bug?
* Can you reproduce the bug using the latest [master](https://github.com/samples-ci/oneAPI-samples) and the latest oneAPI toolkit related to the sample?
* What CPU/GPU, platform, operating system/distribution are you running? The more specific, the better.

## License

The code samples are licensed under the terms in [LICENSE](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.