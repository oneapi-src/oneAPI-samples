from git import Repo
import os
import json
from json.decoder import JSONDecodeError

repo = Repo()

print(repo)
items = [ item.a_path for item in repo.index.diff(None) ]
print(items)

def isCPP(str):
    return str[-4:] == ".cpp" or str[-4:] == ".hpp"

def samplePath(str):
    head, tail = os.path.split(os.path.dirname(str))
    sample = os.path.join(head, "sample.json")
    return sample if os.path.exists(sample) else ""

def readJsonDataFromFile(jsonFile):                 #creating a dictionary
    jsonData = open(jsonFile)           #open the json file
    try: 
        data = json.load(jsonData)      #load json into memory
    except JSONDecodeError as e:
        print(str(e)+': ' + jsonFile)
    return data

def runScrip(fileName):
    json = readJsonDataFromFile(fileName)
    scripts = json["ciTests"]["linux"]
    os.system("cd " + os.path.abspath(os.path.dirname(fileName)))
    os.system("mkdir build")
    os.system("cd build")

    for script in scripts:
        print("running" + script["id"] + "\n")
        for step in script["steps"]:
            print(step)


    os.system("cd " + os.path.abspath(os.path.dirname(fileName)))
    os.system("rm -rf build")




print([isCPP(x) for x in items])
print([samplePath(x) for x in items])

x = [samplePath(x) for x in items if isCPP(x)]
for z in x:
    runScrip(z)

