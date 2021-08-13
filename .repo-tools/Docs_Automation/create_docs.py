from json.decoder import JSONDecodeError
from datetime import date
import os
import json
from collections import OrderedDict
today = date.today()
currentVersion = "2021.4.0"
fileName = "sample.json"
fDeviceTargets = "CODESAMPLESLIST.md"
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

def createHeaders(dataContent):
    nf = open(fDeviceTargets,"w+")
    nf.write(dataContent['mdDeviceTargetIntrop1'] + dataContent['mdDeviceTargetIntrop2'])
    nf.close()
    nf = open(fChangeLogs,"w+")
    nf.write(dataContent['mdChangeLogHeaderp1'])
    nf.close()
    nf = open(freadme,"w+")
    nf.write("## Introduction\n\n")
    nf.write(dataContent['mdIntro1'] + "\n" +dataContent['mdIntro2'] + currentVersion + dataContent['mdIntro2.1'] + "\n ### Sample Details\n\n")
    nf.write(dataContent['mdIntro3'] + dataContent['mdIntro3.1']+dataContent['mdIntro3.2'] + dataContent['mdIntro3.3'] + dataContent['mdIntro3.4'] + dataContent['mdIntro3.5'])
    nf.write("\n" +dataContent['mdIntro4'] + "\n" +dataContent['mdIntro4.1'] + "\n" +dataContent['mdIntro4.2'])
    nf.write("\n\n### On Windows Platform\n\n" + dataContent['mdIntro5.1'] + "\n" + dataContent['mdIntro5.2'] + "\n" + dataContent['mdIntro5.3'] + "\n" + dataContent['mdIntro5.4'])
    nf.write("\n\n## Known Issues or Limitations\n\n## Contributing\n\n" + dataContent['mdIntro6'] + "\n\n" + dataContent['mdIntro7'])
    nf.close()

def createFooters(count):
    #setting up Device Targets
    nf = open(fDeviceTargets,"a")
    nf.write("Total Samples: " + str(count)+ "\n\n")
    nf.write(str(dataContent['mdDeviceTargetFooter']) + d)
    nf.close()
    nf = open(freadme,"a+")
    nf.write(dataContent['mdLicense'])
    nf.close()
    
def openJson(jsonFile):                 #creating a dictionary
    jsonData = open(jsonFile)           #open the json file
    try: 
        data = json.load(jsonData)      #load json into memory
    except JSONDecodeError as e:
        print(str(e)+': ' + jsonFile)
    return data

def readContent():                                  #readin in strings for use in document creation
    jsonFile = '.repo-tools\Docs_Automation\content.json'
    dataContent = openJson(jsonFile)
    return dataContent

def addVersion(dict_main, dict_version):# After walking thu directories we need to add version to dict_main
    for key in dict_version.keys():
        try:
            if (key in dict_main):
                dict_tmp=dict_version[key]  #copy to dict_tmp if true 
                ver=dict_tmp['ver']
                dict_main[key]['ver']=ver

        except KeyError as e: #not working
            print(str(e) + ":No Match for guid: ")
   
def createChangeLog(count): #sorted but does not include version
    temp = dict_main.items()
    sorted_items = sorted(temp, key=lambda key_value: key_value[1]["name"], reverse=False) # sorts by name
    sorted_by_name = OrderedDict(sorted_items)
    temp=sorted_by_name.items()
    sorted_items = sorted(temp, key=lambda key_value: key_value[1]["ver"], reverse=True) # sorts by ver
    sorted_by_name = OrderedDict(sorted_items)
    nf = open(fChangeLogs,"a+")
    nf.write(str(count) + dataContent['mdChangeLogHeaderp2'])
    for key in sorted_by_name.keys():
        description= str(sorted_by_name[key]['description']) 
        url= sorted_by_name[key]['url']
        ver=sorted_by_name[key]['ver']
        name=sorted_by_name[key]['name']
        cat=str(sorted_by_name[key]['categories'])
        if (cat=="['Toolkit/Publication: Data Parallel C++']"):
            description=description.replace('*','<br>')
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)" + "<br><br>[" + name
        nf.write("|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n") 
    nf.close()

def createTtargetedDevices():
    temp = dict_main.items()
    sorted_items = sorted(temp, key=lambda key_value: key_value[1]["name"], reverse=False) # sorts by name
    sorted_by_name = OrderedDict(sorted_items)
    temp=sorted_by_name.items()
    nf = open(fDeviceTargets,"a+")

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
    nf.close()

def createReadme():
    temp = dict_main.items()
    sorted_items = sorted(temp, key=lambda key_value: key_value[1]["name"], reverse=False) # sorts by name
    sorted_by_name = OrderedDict(sorted_items)
    nf = open(freadme,"a+")
    
    for key in sorted_by_name.keys():
        description= str(sorted_by_name[key]['description']) 
        url= sorted_by_name[key]['url']
        ver=sorted_by_name[key]['ver']
        name=sorted_by_name[key]['name']
        cat=str(sorted_by_name[key]['categories'])
        if (cat=="""['Toolkit/Publication: Data Parallel C++']"""):
            name ="Pub: Data Parallel C++:](https://www.apress.com/9781484255735)" + "\n\n[" + name
            description=description.replace('*','<br>')
        if (ver==currentVersion):
            nf.write("|" + ver + "|[" + name+ "](" + url + ")|" + description + "|\n") 
    nf.close()
   
#main
checkFileExists(fDeviceTargets)     #Cleaning up from previous run
checkFileExists(fChangeLogs)        #Cleaning up from previous run
checkFileExists(freadme)            #Cleaning up from previous run
print(absolute_path + " Length: " + str(pathLength) + "\n")
dataContent = readContent()         #read json for data used in creating document header and footers
createHeaders(dataContent)          #create headers for the various documents being generated

dict_main={}                        #initializing Dictionary
dict_version = openJson(fguidVer)

for subdir, dirs, files in os.walk('..\\'):
    for file in files:
        if (file == fileName):
            f = os.path.join(subdir, file)
            fp = os.path.join(subdir)                           #this will be needed to create potential hyperlink
            getPath = fp[pathLength:len(fp)]                    #Generates the path specific path of url
            getPath = getPath.replace('\\','/')                 #char replace \ for /
            fullURL=oneAPIURL+getPath
            data = openJson(f) 
            dict_main[data['guid']]=data   
            dict_main[data['guid']]['url'] = fullURL
            count = count+1

addVersion(dict_main,dict_version)
createChangeLog(count)
createTtargetedDevices()
createReadme()
createFooters(count)
print("Finished")
