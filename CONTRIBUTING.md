# Contributing to oneAPI-samples

The `master` branch contains code samples that work with the latest released version of the [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html). Do not commit change to the `master` branch.

All contributions must go into the `development` branch through a pull request (PR) where they will be reviewed before being merged. At specific dates, corresponding to the releases of the oneapi DPC++/C++ compiler, the `development` branch is merged into the `master` branch.

## Fork the Repository

1. To fork the repository from the GitHub user interface, click the **Fork** icon then select **Create a new fork**. The fork will be created in few seconds. If you previously forked the repo, skip to the Step 5.

2. Select an **Owner** for the forked repository.

3. Deselect the **Copy the master branch only** check box. (It should be unchecked before proceeding to the next step.)

4. Click the **Create fork** button.

5. If you have an existing fork but do not have a `development` branch, create a `development` branch by selecting the oneapi-src/oneAPI-samples `development` branch in the dropdown as the branch source.

6. Once your fork has been created, click the **Settings** icon and find the **Default Branch** section.

7. Click the **Switch to another branch** graphic.

8. From the dropdown, change the default branch to `development`. Click the **Update** button.

9. To create a branch in your fork, make sure the `development` branch is selected from the dropdown, and enter the name of your branch in the text field.

## Clone Your Fork

Clone the repo and checkout the branch that you just created by entering a command similar to the following:

```
git clone -b <your branch name> https://github.com/<your GitHub username>/<your repo name>.git
```

Once you are ready to commit your changes to your repo, enter commands similar to the following:

```
git add .
git commit -s -m "<insert commit reason here>"
git push origin
```

## Submit Pull Requests

When submitting a pull request, keep the following guidelines in mind:

- Make sure that your pull request has a clear purpose; it should be as simple as possible. This approach enables quicker PR reviews.

- Explain anything non-obvious from the code in comments, commit messages, or the PR description, as needed.

- Check the number of files being updated. Ensure that your pull request includes only the files you expected to be changed. (If there are additional files you did not expect included in the commit, troubleshoot before submitting the PR.)

- Never open a pull request to the `master` branch directly, all pull requests must be targeting the `development` branch.

## Log a Bug or Request a Feature

We use [GitHub Issues](https://github.com/oneapi-src/oneAPI-samples/issues) to track sample development issues, bugs, and feature requests.

When reporting a bug, provide the following information when possible:

- Steps to reproduce the bug.
- Whether you found or reproduced the bug using the latest sample in the `master` branch and the latest Intel® oneAPI Toolkits.
- Version numbers or other information about the CPU/GPU/FPGA/device, platform, operating system or distribution you used to find the bug.

For usage, installation, or other requests for help, go to the [Intel® oneAPI Forums](https://software.intel.com/en-us/forums/intel-oneapi-forums) for more information.

## License

Code samples in this repository are licensed under the terms outlined in [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.