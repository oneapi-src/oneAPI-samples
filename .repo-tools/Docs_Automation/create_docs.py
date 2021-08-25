from json.decoder import JSONDecodeError
from pathlib import Path
from datetime import date
import os
import json
from collections import OrderedDict
today = date.today()
currentVersion = "2021.4.0"
fileName = "sample.json"
fCodeSamplesLists = "CODESAMPLESLIST.md"
fChangeLogs = "CHANGELOGS.md"
freadme = "README.md"
fguidVer = ".repo-tools\Docs_Automation\guids.json"
absolute_path = os.getcwd()
pathLength = len(absolute_path)   #used to modify local path length, so only tree structure is left for crafting the url
oneAPIURL = 'https://github.com/oneapi-src/oneAPI-samples/tree/master'
d = today.strftime("%B %d, %Y")
count = 0

def checkFileExists(checkFile):
    if os.path.exists(checkFile):
        os.remove(checkFile)
    else:
        print("The " + checkFile + " file does not exist")

def openJson(jsonFile):                 #creating a dictionary
    jsonData = open(jsonFile)           #open the json file
    try: 
        data = json.load(jsonData)      #load json into memory
    except JSONDecodeError as e:
        print(str(e)+': ' + jsonFile)
    return data

def readContent():                      #reading in strings for use in document creation
    jsonFile = '.repo-tools\Docs_Automation\content.json'
    dataContent = openJson(jsonFile)
    return dataContent
   
def createChangeLog(count,sorted_by_name,sorted_by_ver): #sorted but does not include version
    nf = open(fChangeLogs,"w+")
    nf.write(dataContent['mdChangeLogHeaderp1'])
    nf.write(dataContent['mdChangeLogHeaderp2'])
  
    for key in sorted_by_ver.keys():
        description= str(sorted_by_name[key]['description']) 
        url= sorted_by_name[key]['url']
        ver=sorted_by_ver[key]['ver']
        name=sorted_by_name[key]['name']
        cat=str(sorted_by_name[key]['categories'])

        # Due to name issues, we need to fix the DPC** books chapter namesas its found and put it into the doc
        if (cat=="['Toolkit/Publication: Data Parallel C++']"):
            description=description.replace('*','<br>')
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)" + "<br><br>[" + name
        
        nf.write("|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n") 
    nf.write("Total Samples: " + str(count)+ "\n\n")
    nf.write(str(dataContent['mdCodeSamplesListFooter']) + d)
    nf.close()
    print("Change Log has been created")

def createCodeSamplesList():
    temp = dict_main.items()
    sorted_items = sorted(temp, key=lambda key_value: key_value[1]["name"], reverse=False) # sorts by name
    sorted_by_name = OrderedDict(sorted_items)
    temp=sorted_by_name.items()
    #nf = open(fCodeSamplesLists,"a+")
    nf = open(fCodeSamplesLists,"w+")
    nf.write(dataContent['mdCodeSamplesListIntrop1'] + dataContent['mdCodeSamplesListIntrop2'])
    for key in sorted_by_name.keys():
        description= str(sorted_by_name[key]['description']) 
        url= sorted_by_name[key]['url']
        name=sorted_by_name[key]['name']
        target= str(sorted_by_name[key]['targetDevice'])
        cat=str(sorted_by_name[key]['categories'])
        if (cat=="""['Toolkit/Publication: Data Parallel C++']"""):
            description=description.replace('*','<br>')
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)" + "<br><br>[" + name
        nf.write("|[" + name+ "](" + url + ")|" + target + "|" + description + "|\n") 
    
    nf.write("Total Samples: " + str(count)+ "\n\n")
    nf.write(str(dataContent['mdCodeSamplesListFooter']) + d)
    nf.write("Total Samples: " + str(count)+ "\n\n")
    nf.close()
    print("Code Samples List has been created")

def createReadme(sorted_by_name, sorted_by_ver):
    nf = open(freadme,"w+")
    nf.write("## Introduction\n\n")
    nf.write(dataContent['mdIntro1'] + "\n" +dataContent['mdIntro2'] + currentVersion + dataContent['mdIntro2.1'] + "\n ### Sample Details\n\n")
    nf.write(dataContent['mdIntro3'] + dataContent['mdIntro3.1']+dataContent['mdIntro3.2'] + dataContent['mdIntro3.3'] + dataContent['mdIntro3.4'] + dataContent['mdIntro3.5'])
    nf.write("\n" +dataContent['mdIntro4'] + "\n" +dataContent['mdIntro4.1'] + "\n" +dataContent['mdIntro4.2'])
    nf.write("\n\n### On Windows Platform\n\n" + dataContent['mdIntro5.1'] + "\n" + dataContent['mdIntro5.2'] + "\n" + dataContent['mdIntro5.3'] + "\n" + dataContent['mdIntro5.4'])
    nf.write("\n\n## Known Issues or Limitations\n\n## Contributing\n\n" + dataContent['mdIntro6'] + "\n\n" + dataContent['mdIntro7'])
        
    for key in sorted_by_name.keys():
        description= str(sorted_by_name[key]['description']) 
        url= sorted_by_name[key]['url']
        ver=sorted_by_ver[key]['ver']
        name=sorted_by_name[key]['name']
        cat=str(sorted_by_name[key]['categories'])
        if (cat=="""['Toolkit/Publication: Data Parallel C++']"""):
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)" + "\n\n[" + name
            description=description.replace('*','<br>')
        if (ver==currentVersion):
            nf.write("|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n") 
    nf.write("Total Samples: " + str(count)+ "\n\n")
    nf.write(dataContent['mdLicense'])
    nf.write("\n\n")
    nf.write(str(dataContent['mdCodeSamplesListFooter']) + d)
    nf.close()
    print("Readme has been created")

#main
checkFileExists(fCodeSamplesLists)     #Cleaning up from previous run
checkFileExists(fChangeLogs)        #Cleaning up from previous run
checkFileExists(freadme)            #Cleaning up from previous run
dataContent = readContent()         #read json for data used in creating document header and footers

dict_main={}                        
dict_version = openJson(fguidVer)

for subdir, dirs, files in os.walk('..\\'):  # walk through samples repo looking for samples.json, if found add data to dict_main
    for file in files:
        if (file == fileName):
            f = os.path.join(subdir, file)
            data = openJson(f) 
            dict_main[data['guid']]=data   
            # build url
            fp = os.path.join(subdir)
            fp = fp.replace('\\','/')                 #char replace \ for /
            fullURL=oneAPIURL+(str(fp)[17:])  # removed first 17 characters of the path, which is always "../oneAPI-samples/"
            #end build url
            dict_main[data['guid']]['url'] = fullURL
            count = count+1

temp = dict_main.items()
sorted_by_name = OrderedDict(sorted(temp, key=lambda key_value: key_value[1]["name"], reverse=False))

temp=dict_version.items()
sorted_by_ver = OrderedDict(sorted(temp, key=lambda key_value: key_value[1]["ver"], reverse=True))

createChangeLog(count,sorted_by_name,sorted_by_ver)
createCodeSamplesList()
createReadme(sorted_by_name,sorted_by_ver)

print("Finished")
