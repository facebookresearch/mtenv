# Contributing to MTEnv

We are glad that you want to contribute to MTEnv.

## Local Setup

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


### Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation.
Follow the steps to build/update the documentation locally.

* rm -rf docs/*
* rm -rf docs_src/source/pages/api
* rm -rf docs_src/build
* sphinx-apidoc -o docs_src/source/pages/api mtenv
* cd docs_src
* make html
* cd ..
* cp -r docs_src/build/html/* docs/

Or run all the commands at once: `rm -rf docs/* && rm -rf docs_src/source/pages/api && rm -rf docs_src/build && sphinx-apidoc -o docs_src/source/pages/api mtenv && cd docs_src && make html && cd .. && cp -r docs_src/build/html/* docs/`

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Set up the code using instructions from above.
3. If you are adding a new environment, checkout the guide on [how to contribute new environments](#How-To-Contribute-New-Environments). 
4. If you've added code that should be tested, add tests. 
5. If you've changed APIs, update the documentation.
6. Ensure the test suite passes. This is tested via CI when you make a PR.
7. Add a news entry as described [here](#News-Entry).
8. If you haven't already, complete the Contributor License Agreement ("CLA").

#### How To Contribute New Environments

1. We recommend that you first open an issue to discuss the feasibility of 
adding a new environment. This will eliminate the possibility of duplication 
of work.
2. Checkout the guide on [how to create new environments](https://mtenv.readthedocs.io/en/latest/pages/envs/create.html).
3. Create a new folder in `mtenv/envs`.
4. Add the following files, along with the implementation of the environment.
You can refer to existing environments.
    * `__init__.py`
    * `setup.py`
    * `requirements.txt`
    * `README.md`
5. Register your environment in `/mtenv/envs/__init__.py`. 
    * `test_kwargs` are optional but if you can specify some values (both 
    valid and invalid configurations) for automated testing.
6. We run some basic tests on the environment (to make sure it can be 
instantiated). You should add more tests to `tests/envs`
7. Add your environment to the list of supported environments at 
`docs_src/source/pages/envs/supported.rst`

#### News Entry

* Add an issue describing the issue that the PR fixes.

* Create a file, with the name `issue_number.xxx`, in `news` folder using 
the issue number from the previous step.

* The extension (ie `xxx` part) can be one of the following:

    * api_change: API Changes
    * bugfix: Bug Fixes
    * doc: Documentation Changes
    * environment: Environment Chages (addition or removal)
    * feature: Features
    * misc: Miscellaneous Changes

* Add a crisp one line summary of the change. The summary should complete
the sentence "This change will ...". 

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