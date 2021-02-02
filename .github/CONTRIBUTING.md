# Contributing to MTEnv

We are glad that you want to contribute to MTEnv.

### Local Setup
Follow these instructions to setup MTEnv locally:

* Clone locally - `git clone git@github.com:facebookresearch/mtenv.git`.
* *cd* into the directory - `cd mtenv`.
* Install MTEnv in the dev mode - `pip install -e ".[dev]"`
* Tests can be run locally using `nox`. The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* Install pre-commit hooks - `pre-commit install`. It will execute some
of the tests when you commit the code. You can disable it by adding the
"-n" flag to git command. For example, `git commit -m <commit_message> -n`.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. Set up the code using instructions from above.
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the documentation.
5. Ensure the test suite passes.
7. Add a news entry.
8. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to MTEnv, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.