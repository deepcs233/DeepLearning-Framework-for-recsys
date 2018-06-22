from conf import *

with open(PROCESSED_TRAIN_DATA_PATH) as f:
    data = f.readlines()

fids = set([])

for line in data[1:]:
    ems = line.strip().split()
    for t in ems[1:]:
        fids.add(t)

print len(fids)
