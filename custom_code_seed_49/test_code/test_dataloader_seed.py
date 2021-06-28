###############
# This code is just to ensure that the seeding functionality is working properly in the dataloader or not
###############

from dataloader_ import KvasirDataset, FedDataset
import time
import numpy as np

t1 = FedDataset(KvasirDataset, seed=49)
t2 = FedDataset(KvasirDataset, seed=49)

t1._delayed_init("1,2")
t2._delayed_init("2,2")

l1 = len(t1.train_set.images_names)
l2 = len(t2.train_set.images_names)

print(f"number of images in t1 : {l1}")
print(f"number of images in t2 : {l2}")

ran_nums = [159, 143,   9, 147, 132, 367, 206, 366, 268,  27,  64, 310, 359, 127,  92,  67, 102, 368, 197,  35]

s1 = "t1\n"
for i in range(len(ran_nums)):
    s1 += f"{t1.train_set.images_names[i]}\n"

s2 = "t2\n"
for i in range(len(ran_nums)):
    s2 += f"{t2.train_set.images_names[i]}\n"

curr_timestamp = time.time()

with open(f"run_t1_{curr_timestamp}.txt","w") as f:
    f.write(s1)

with open(f"run_t2_{curr_timestamp}.txt","w") as f:
    f.write(s2)