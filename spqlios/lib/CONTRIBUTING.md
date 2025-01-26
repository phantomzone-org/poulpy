# Contributing to SPQlios-fft

The spqlios-fft team encourages contributions.
We encourage users to fix bugs, improve the documentation, write tests and to enhance the code, or ask for new features.
We encourage researchers to contribute with implementations of their FFT or NTT algorithms.
In the following we are trying to give some guidance on how to contribute effectively.

## Communication ##

Communication in the spqlios-fft project happens mainly on [GitHub](https://github.com/tfhe/spqlios-fft/issues).

All communications are public, so please make sure to maintain professional behaviour in
all published comments. See [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) for
guidelines.

## Reporting Bugs or Requesting features ##

Bug should be filed at [https://github.com/tfhe/spqlios-fft/issues](https://github.com/tfhe/spqlios-fft/issues).

Features can also be requested there, in this case, please ensure that the features you request are self-contained,
easy to define, and generic enough to be used in different use-cases. Please provide an example of use-cases if
possible.

## Setting up topic branches and generating pull requests

This section applies to people that already have write access to the repository. Specific instructions for pull-requests
from public forks will be given later.

To implement some changes, please follow these steps:

- Create a "topic branch". Usually, the branch name should be `username/small-title`
  or better `username/issuenumber-small-title` where `issuenumber` is the number of
  the github issue number that is tackled.
- Push any needed commits to your branch. Make sure it compiles in `CMAKE_BUILD_TYPE=Debug` and `=Release`, with `-DWARNING_PARANOID=ON`.
- When the branch is nearly ready for review, please open a pull request, and add the label `check-on-arm`
- Do as many commits as necessary until all CI checks pass and all PR comments have been resolved.

  >  _During the process, you may optionnally use `git rebase -i` to clean up your commit history. If you elect to do so,
  please at the very least make sure that nobody else is working or has forked from your branch: the conflicts it would generate
  and the human hours to fix them are not worth it. `Git merge` remains the preferred option._

- Finally, when all reviews are positive and all CI checks pass, you may merge your branch via the github webpage.

### Keep your pull requests limited to a single issue

Pull requests should be as small/atomic as possible.

### Coding Conventions

* Please make sure that your code is formatted according to the `.clang-format` file and
  that all files end with a newline character.
* Please make sure that all the functions declared in the public api have relevant doxygen comments.
  Preferably, functions in the private apis should also contain a brief doxygen description.

### Versions and History

* **Stable API** The project uses semantic versioning on the functions that are listed as `stable` in the documentation. A version has
  the form `x.y.z`
    * a patch release that increments `z` does not modify the stable API.
    * a minor release that increments `y` adds a new feature to the stable API.
    * In the unlikely case where we need to change or remove a feature, we will trigger a major release that
    increments `x`.

    >    _If any, we will mark those features as deprecated at least six months before the major release._

* **Experimental API** Features that are not part of the stable section in the documentation are experimental features: you may test them at
  your own risk,
  but keep in mind that semantic versioning does not apply to them.

>  _If you have a use-case that uses an experimental feature, we encourage
>  you to tell us about it, so that this feature reaches to the stable section faster!_

* **Version history** The current version is reported in `manifest.yaml`, any change of version comes up with a tag on the main branch, and the history between releases is summarized in `Changelog.md`. It is the main source of truth for anyone who wishes to
  get insight about
  the history of the repository (not the commit graph).

>  Note: _The commit graph of git is for git's internal use only. Its main purpose is to reduce potential merge conflicts to a minimum, even in scenario where multiple features are developped in parallel: it may therefore be non-linear. If, as humans, we like to see a linear history, please read `Changelog.md` instead!_
