#! /usr/bin/python
with open("redcarmatchesout.txt", 'r') as f:
    text = f.read()
    items=[]
    for line in text.split('\n'):
     if line:
        id = int(line)
        match = False
        for item in items:
           if item < id and id-item < 1000:
              match = True
              break
           if item > id and item-id < 1000:
              match = True
              break
        if match == False:
           items+=[id]
    for item in items:
        print(f"{item:06d}")