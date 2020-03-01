
def readData(filename):
	fr = open(filename)
	returnData = []
	headerLine=fr.readline()###move cursor 
	for line in fr.readlines():
		lineStrip = line.strip().replace('"','')
		lineList =	lineStrip.split('\t')
		returnData.append(lineList)###['3','2',...]
	return returnData
  
  
  
def readData2(filename):
	fr = open(filename)
	returnData = []
	headerLine=fr.readline()###move cursor 
	for line in fr.readlines():
		linestr = line.strip().replace(', ','')
		lineList =	list(linestr)
		returnData.append(lineList)###['3','2',...]
	return returnData
  
def readData3(filename):
	fr = open(filename)
	returnData = []
	for line in fr.readlines():
		lineList = line.strip().split(',')
		returnData.append(lineList)###['3','2',...]
	return returnData

nurse = readData3("data folder/nursery.data")
nurse.pop()
nurse = np.array(nurse) 
	
	
	
def readarff(filename):
  arrfFile = open(filename)
  lines = [line.rstrip('\n') for line in arrfFile]
  data = [[]]
  index = 0
  for line in lines :
      if(line.startswith('@attribute')) :
          index+=1
      elif(not line.startswith('@data') and not line.startswith('@relation') and not line.startswith('%')) :
          data.append(line.split(','))
  del data[0]
  return data
