# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue) 

## Type of change

Please delete options that are not relevant. Add a 'X' to the one that is applicable. 

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Sample Migration (Moving sample from old repository after completing checklist established)

# How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce. Please also list any relevant details for your test configuration

- [ ] Command Line
- [ ] oneapi-cli
- [ ] Visual Studio
- [ ] Eclipse IDE
- [ ] VSCode

# Checklist for Moving samples:
Links and Details can be found in the samples WG Teams Files. 

- [ ] Review sample design with domain reviewers https://github.com/oneapi-src/oneAPI-samples/wiki/Reviewers-and-Domain-Experts 
- [ ] Implement coding guidelines and ensure code quality.
- [ ] Adhere to sample.json specification. https://github.com/oneapi-src/oneAPI-samples/wiki/sample-json-specification
- [ ] Run jsonlint on sample.json to verify json syntax. www.jsonlint.com
- [ ] Adhere to readme template 
- [ ] Ensure/create CI test configurations for sample (ciTests field) https://github.com/oneapi-src/oneAPI-samples/wiki/sample-json-ci-test-object
- [ ] Enforce format via clang-format config file
- [ ] Compile code using compiler flags and fix anything detected "enable /Wall -Wformat-security -Werror=format-security"
- [ ] Review Sample with Domain Expert: <insert NameHere>
- [ ] Review DPC++ code with Paul Peterseon. (GitHub User: pmpeter1)
- 
 ] Review readme with Tom Lenth or Joe Oster. (GitHub User: JoeOster)
- [ ] Tested using Dev Cloud when applicable
- [ ] Implement fixes for ONSAM Jiras
- [ ] If you have new dependencies/binaries, inform Samples Project Manager Swapna R Dontharaju (@srdontha) or @JoeOster

