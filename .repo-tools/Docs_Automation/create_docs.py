from datetime import date
from collections import OrderedDict
from json.decoder import JSONDecodeError
from pathlib import Path
import os
import json
today = date.today()
d = today.strftime("%B %d, %Y")
currentVersion = "2021.4.0"
fileName = "sample.json"
fCodeSamplesLists = "CODESAMPLESLIST.md"
fChangeLogs = "CHANGELOGS.md"
freadme = "README.md"
fguidVer = ".repo-tools\Docs_Automation\guids.json"
#fguidVer = ".repo-tools/Docs_Automation/guids.json"  #for use when linux is supported for this tool
oneAPIURL = 'https://github.com/oneapi-src/oneAPI-samples/tree/master'
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
    #jsonFile = '.repo-tools/Docs_Automation/content.json' #for use when linux is supported for this tool
    jsonFile = '.repo-tools\Docs_Automation\content.json'
    dataContent = openJson(jsonFile)
    return dataContent
   
def createChangeLog(count,sorted_by_name,sorted_by_ver): #sorted but does not include version
    nf = open(fChangeLogs,"w+")
    nf.write(dataContent['mdChangeLogHeaderp1'])
    nf.write(dataContent['mdChangeLogHeaderp2'])
  
    for key in sorted_by_ver.keys():
        try:
            description= str(sorted_by_name[key]['description']) 
            url= sorted_by_name[key]['url']
            name=sorted_by_name[key]['name']
            cat=str(sorted_by_name[key]['categories'])
        except KeyError as e:
            print("Error with: "+key+ "Missing from guids.json")  
        ver=sorted_by_ver[key]['ver']
        
        # Due to name issues, we need to fix the DPC** books chapter namesas its found and put it into the doc
        if (cat=="['Toolkit/Publication: Data Parallel C++']"):
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)<br>[" + name
            description=description.replace('*','')
            description=description.replace('fig_','<br>- Fig_')
            description="Collection of Code samples for the chapter"+description
        
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
    nf = open(fCodeSamplesLists,"w+")
    nf.write(dataContent['mdCodeSamplesListIntrop1'] + "\n\n" + dataContent['mdCodeSamplesListIntrop2'])
    for key in sorted_by_name.keys():
        description= str(sorted_by_name[key]['description']) 
        url= sorted_by_name[key]['url']
        name=sorted_by_name[key]['name']
        target= str(sorted_by_name[key]['targetDevice'])
        cat=str(sorted_by_name[key]['categories'])
        if (cat=="""['Toolkit/Publication: Data Parallel C++']"""):
            description=description.replace('*','')
            description=description.replace('fig_','<br>- Fig_')
            description="Collection of Code samples for the chapter"+description
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)<br>[" + name
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
    nf.write(dataContent['mdIntro5'] + dataContent['mdIntro5.1'] + "\n" + dataContent['mdIntro5.2'] + "\n" + dataContent['mdIntro5.3'] + "\n" + dataContent['mdIntro5.4'])
    nf.write("\n\n" +dataContent['mdIntro4'])
    nf.write(dataContent['mdIntro6'] + "\n\n" + dataContent['mdIntro7'])
        
    for key in sorted_by_name.keys():
        try:
            description= str(sorted_by_name[key]['description']) 
            url= sorted_by_name[key]['url']
            name=sorted_by_name[key]['name']
            cat=str(sorted_by_name[key]['categories'])
            ver=sorted_by_ver[key]['ver']
            
        except KeyError as e:
            print("Error with: "+key)  

        if (cat=="""['Toolkit/Publication: Data Parallel C++']"""):
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)<br>[" + name
            description=description.replace('*','')
            description=description.replace('fig_','<br>- Fig_')
            description="Collection of Code samples for the chapter"+description
        if (ver==currentVersion):
            nf.write("|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n") 
    nf.write("\nTotal Samples: " + str(count)+ "\n\n")
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
        # Future - add a search for license file and if none, show a warning
        # Future - if no sample.json is present then show a warning
        # future - if no readme.md is present then show a warning
        # future - Check dict_main vs dict_version for guid present if not then need to add
        # furure - check dict_version vs dict_main for guid present if not then need to allow if new sample hasnt been uploaded
        # Future - for readme, need to add what samples may have been removed for this "current version" 

createChangeLog(count,sorted_by_name,sorted_by_ver)
createCodeSamplesList()
createReadme(sorted_by_name,sorted_by_ver)

print("Finished")
