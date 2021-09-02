## Description

Adding automation for development of various documents, this will elimate making multiple changes per release cycle, effectively making the process less error prone. Tool currently works on Windows* only.

## Running the tool
	1. Update your fork to latest from oneapi-src/oneAPI-samples
	2. Create a branch ex: (New Docs2021.4)to hold the new docs changes
	3. Switch Branch navigate to the new Branch
	4. Update guids.json and content.json as appropriate
	5. From the root of the repo, run .repo-tools/Docs_Automation/create_docs.py
	6. This will generate the three files in the repo root. CODESMPLESLIST.md, CHNGELOGS.md, README.md

## File List
|Files |location|Descriptions|
|---|-|--|
|content.json|.repo-tools/Doc_Automation|This is a json containing various strings to create the documents. Any doc changes need to be made here|
|create_docs.py|.repo-tools/Doc_Automation|Tool that creates thefiles listed below. |
|guids.json|.repo-tools/Doc_Automation|List of guids, names and versions. This file needs to be updated if new samples are added|
|Generated Files |
|CODESAMPLESLIST.md| repo root|List of code samples in alphabetical order with targeted device and description of each sample |
|CHANGELOGS.md| repo root|A running list of all samples sorted by Version(newest) and Alphabetical|
|README.md| repo root|This is the introductory readme at the root of the repo |
