from datetime import date
from collections import OrderedDict
from json.decoder import JSONDecodeError
from pathlib import Path
import os
import json

today = date.today()
d = today.strftime("%B %d, %Y")
currentVersion = "2022.1.0"
fCodeSamplesLists = "CODESAMPLESLIST.md"
fChangeLogs = "CHANGELOGS.md"
freadme = "README.md"
guidsVerPath = os.path.join('.repo-tools', 'Docs_Automation', 'guids.json')
contentPath = os.path.join('.repo-tools', 'Docs_Automation', 'content.json')
bookPath = os.path.join('Publications','DPC++')
oneAPIURL = 'https://github.com/oneapi-src/oneAPI-samples/tree/master'

def removeIfFileExist(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)
    else:
        print("The " + fileName + " file does not exist")

def readJsonDataFromFile(jsonFile):                 #creating a dictionary
    jsonData = open(jsonFile)           #open the json file
    try: 
        data = json.load(jsonData)      #load json into memory
    except JSONDecodeError as e:
        print(str(e)+': ' + jsonFile)
    return data

def createChangeLogFillerText(textFillers, count):
    header = textFillers['mdChangeLogHeaderp1'] + textFillers['mdChangeLogHeaderp2']
    footer = "Total Samples: " + str(count)+ "\n\n" + textFillers['mdCodeSamplesListFooter'] + d + "\n"
    return header, footer

def createCodeSamplesListFillerText(textFillers, count):
    header = textFillers['mdCodeSamplesListIntrop1'] + "\n\n" + textFillers['mdCodeSamplesListIntrop2']
    footer = "Total Samples: " + str(count)+ "\n\n" + str(textFillers['mdCodeSamplesListFooter']) + d + "\n"
    return header, footer

def createReadMeFillerText(textFillers, count):
    header = ("## Introduction\n\n"
        + textFillers['mdIntro1'] + "\n" +textFillers['mdIntro2'] + currentVersion + textFillers['mdIntro2.1'] + "\n ### Sample Details\n\n"
        + textFillers['mdIntro3'] + textFillers['mdIntro3.1']+textFillers['mdIntro3.2'] + textFillers['mdIntro3.3'] + textFillers['mdIntro3.4'] + textFillers['mdIntro3.5']
        + textFillers['mdIntro5'] + textFillers['mdIntro5.1'] + "\n" + textFillers['mdIntro5.2'] + "\n" + textFillers['mdIntro5.3'] + "\n" + textFillers['mdIntro5.4']
        + "\n\n" +textFillers['mdIntro4'] + textFillers['mdIntro6'] + "\n\n" + textFillers['mdIntro7'] )
    
    removedSamplesHeader = ( "\nTotal Samples: " + str(count) + "\n"
        + textFillers["mdDeletedSample"]
    )

    footer = ('\n\n' + textFillers['mdLicense']
        + "\n\n" + str(textFillers['mdCodeSamplesListFooter']) + d + "\n")
    return header, removedSamplesHeader, footer

def replaceIfBook(cat, name, description):
    # Due to name issues, we need to fix the DPC** books chapter namesas its found and put it into the doc
    if (cat == "['Toolkit/Publication: Data Parallel C++']"):
        name = "Pub: Data Parallel C++:](https://www.apress.com/9781484255735)<br>[" + name
        description = description.replace('*', '')
        description = description.replace('fig_', '<br>- Fig_')
        description = "Collection of Code samples for the chapter" + description

    return name, description

def generateChangeLogLines(currentData, guidsVersions):
    for key in guidsVersions.keys():
        try:
            description = currentData[key]['description']
            url = currentData[key]['url']
            name = currentData[key]['name']
            cat = str(currentData[key]['categories'])
        except KeyError as e:
            print("\tWarning with: " + key + " Missing from guids.json\n\t\t" + guidsVersions[key]['notes'][:50] + "...")
            continue
        ver = guidsVersions[key]['ver']

        name, description = replaceIfBook(cat, name, description)

        yield "|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n"

def generateCodeSamplesListLines(currentData):
    for item in currentData.values():
        description = item['description'] 
        url = item['url']
        name = item['name']
        target = str(item['targetDevice'])
        cat = str(item['categories'])
        
        name, description = replaceIfBook(cat, name, description)

        yield "|[" + name+ "](" + url + ")|" + target + "|" + description + "|\n"

def generateReadmeLines(currentData, guidsVersions):
    for key, item in currentData.items():
        try:
            description= item['description']
            url = item['url']
            name =item['name']
            cat=str(item['categories'])
            ver=guidsVersions[key]['ver']
        except KeyError as e:
            print("\tError with: " + key + " Missing in guids.json")  

        name, description = replaceIfBook(cat, name, description)
        
        if (ver == currentVersion):
            yield ("|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n") 

def generateReadmeDeleted(guidsVersions):
    for item in guidsVersions.values():
        if item['removed'] != "False":
            yield (f"| {item['ver']} | {item['removed']} | {item['name']} | {item['notes']} |"
                + f" [{item['removed']}](https://github.com/oneapi-src/oneAPI-samples/releases/tag/{item['removed']})" 
                + (f" Path: {item['path']}" if 'path' in item else '')
                + "|\n")

def createChangeLog(textFillers, count, currentData, guidsVersions): #sorted but does not include version
    print("Creating ChangeLog started")
    header, footer = createChangeLogFillerText(textFillers, count)
    with open(fChangeLogs,"w+") as file:
        file.write(header)
        for line in generateChangeLogLines(currentData, guidsVersions):        
            file.write(line) 
        file.write(footer)
    print("ChangeLog has been created")

def createCodeSamplesList(currentData, textFillers, count):
    print("Creating CodeSamplesList started")
    header, footer = createCodeSamplesListFillerText(textFillers, count)
    with open(fCodeSamplesLists,"w+") as file:
        file.write(header)
        for line in generateCodeSamplesListLines(currentData):
            file.write(line) 
        file.write(footer)
    print("Code Samples List has been created")

def createReadme(textFillers, count, currentData, guidsVersions):
    print("Creating ReadMe started")
    header, removedSamplesHeader, footer = createReadMeFillerText(textFillers, count)
    with open(freadme,"w+") as file:
        file.write(header)    
        for line in generateReadmeLines(currentData, guidsVersions):
            file.write(line)

        file.write(removedSamplesHeader)
        for line in generateReadmeDeleted(guidsVersions):
            file.write(line)

        file.write(footer)
        print("Readme has been created")

def logIfLicenseNotExists(dir):
    if Path(bookPath) in Path(dir).parents:
        return
    if not( os.path.exists(os.path.join(dir, "License.txt")) or
            os.path.exists(os.path.join(dir, "license.txt")) or
            os.path.exists(os.path.join(dir, "LICENSE.txt"))):
        for _1, _2, files in os.walk(dir): # Additional check if subdirs contains License e.g. \DirectProgramming\DPC++\Jupyter\oneapi-essentials-training
            for file in files:
                if(file == "License.txt"):
                    return
        print("Warning, License.txt not exists in: " + dir)

def logIfReadmeNotExists(dir):
    if Path(bookPath) in Path(dir).parents:
        return
    if not( os.path.exists(os.path.join(dir, "Readme.md")) or
            os.path.exists(os.path.join(dir, "README.md")) or
            os.path.exists(os.path.join(dir, "readme.md"))):
        print("Warning, README.md not exists in: " + dir)

def createNewGuidRecord(data):
    print(f"Adding new GUID to guid.json: {data['guid']}")
    return {'guid':data['guid'], 'ver': currentVersion, 'name':data['name'], 'notes':'-', 'removed':'False'}

def updateGuids(guidsVersions):
    with open(guidsVerPath, "w") as file:
        json.dump(guidsVersions, file, indent=4, sort_keys=True)
    

def main():
    removeIfFileExist(fCodeSamplesLists)     #Cleaning up from previous run
    removeIfFileExist(fChangeLogs)        #Cleaning up from previous run
    removeIfFileExist(freadme)            #Cleaning up from previous run
    textFillers = readJsonDataFromFile(contentPath)         #read json for data used in creating document header and footers
    guidsVersions = readJsonDataFromFile(guidsVerPath)
    currentData={}                        
    count = 0
    guidsNeedsUpdate = False

    for subdir, dirs, files in os.walk('.'):  # walk through samples repo looking for samples.json, if found add data to currentData
        for file in files:
            if (file == 'sample.json'):
                pathToFile = os.path.join(subdir, file)
                data = readJsonDataFromFile(pathToFile) 
                currentData[data['guid']]=data                  
                fullURL = oneAPIURL + subdir[1:].replace('\\', '/') # removed first character which points to current directory "."
                currentData[data['guid']]['url'] = fullURL
                count += 1

                if data['guid'] not in guidsVersions: #Check if new sample is added and update guids.json if necessary
                    guidsVersions[data['guid']] = createNewGuidRecord(data)
                    guidsNeedsUpdate = True

                logIfLicenseNotExists(subdir)
                logIfReadmeNotExists(subdir)


    currentData = OrderedDict(sorted(currentData.items(), key=lambda key_value: key_value[1]["name"].lower()))
    #stable sort so, sort by names first then by version 
    temp = sorted(guidsVersions.items(), key=lambda key_value: key_value[1]['name'].lower())
    guidsVersions = OrderedDict(sorted(temp, key=lambda key_value: key_value[1]["ver"], reverse=True))

    if guidsNeedsUpdate:
        updateGuids(guidsVersions)

    createChangeLog(textFillers, count, currentData, guidsVersions)
    createCodeSamplesList(currentData, textFillers, count)
    createReadme(textFillers, count, currentData, guidsVersions)

    # print("Finished")


if __name__ == "__main__":
    main()


