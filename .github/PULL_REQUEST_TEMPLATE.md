# Existing Sample Changes
## Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes Issue# 

## External Dependencies

List any external dependencies created as a result of this change.

## Type of change

Please delete options that are not relevant. Add a 'X' to the one that is applicable. 

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Implement fixes for ONSAM Jiras

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce. Please also list any relevant details for your test configuration

- [ ] Command Line
- [ ] oneapi-cli
- [ ] Visual Studio
- [ ] Eclipse IDE
- [ ] VSCode
- [ ] When compiling the compliler flag "-Wall -Wformat-security -Werror=format-security" was used

**_Delete this line and everything below if this is not a PR for a new code sample_**

**_Delete this line and all above it if this PR is for a new code sample_**
# Adding a New Sample(s)

## Description

Please include a description of the sample

## Checklist
Administrative
- [ ] Review sample design with the appropriate [Domain Expert](https://github.com/oneapi-src/oneAPI-samples/wiki/Reviewers-and-Domain-Experts): <insert Name Here>
- [ ] If you have any new dependencies/binaries, inform the oneAPI Code Samples Project Manager: @JoeOster

Code Development
- [ ] Implement coding guidelines and ensure code quality. [see wiki for details](https://github.com/oneapi-src/oneAPI-samples/wiki/General-Code-Guidelines)
- [ ] Adhere to readme template 
- [ ] Enforce format via clang-format config file
- [ ] Adhere to sample.json specification. https://github.com/oneapi-src/oneAPI-samples/wiki/sample-json-specification
- [ ] Ensure/create CI test configurations for sample (ciTests field) https://github.com/oneapi-src/oneAPI-samples/wiki/sample-json-ci-test-object
- [ ] Run jsonlint on sample.json to verify json syntax. www.jsonlint.com

Security and Legal
- [ ] OSPDT Approval (see @JoeOster for assistance)
- [ ] Compile using the following compiler flags and fix any warnings, the falgs are: "/Wall -Wformat-security -Werror=format-security"
- [ ] Bandit Scans (Python only)
- [ ] Virus scan

Review
- [ ] Review DPC++ code with Paul Peterseon. (GitHub User: pmpeter1)
- [ ] Review readme with Tom Lenth(@tomlenth) and/or Joe Oster(@JoeOster)
- [ ] Tested using Dev Cloud when applicable
