### STEP 3: UCF-101 split ###
# Script usato per splittare UCF-101 in test, train e val.
### --------------------------------------- ###

import os
import argparse
import numpy as np
import shutil

args = {"source": "../datasets/UCF-101", "dest": "../datasets/UCF-101-splitted", "val_size": 0.1, "train_size": 0.7, "test_size": 0.2}

# Crea le cartelle di destinazione (se non esistono)
if not os.path.exists(args["dest"]):
    os.makedirs(args["dest"])

dest_train = os.path.join(args["dest"], "train")
if not os.path.exists(dest_train):
    os.makedirs(dest_train)

dest_test = os.path.join(args["dest"], "test")
if not os.path.exists(dest_test):
    os.makedirs(dest_test)

dest_val = os.path.join(args["dest"], "val")
if not os.path.exists(dest_val):
    os.makedirs(dest_val)

class_dirs = os.listdir(args["source"])
for class_dir in class_dirs:
    class_dir_path = os.path.join(args["source"], class_dir)
    num_elem = len(os.listdir(class_dir_path))
    total_indices = np.arange(num_elem)
    val_indices = np.random.choice(total_indices,
                                   size=int(num_elem*args["val_size"]),
                                   replace=False)
    total_indices = np.delete(total_indices, val_indices)
    test_indices = np.random.choice(total_indices,
                                   size=int(num_elem*args["test_size"]),
                                   replace=False)

    if not os.path.exists(os.path.join(dest_train, class_dir)):
        os.mkdir(os.path.join(dest_train, class_dir))
    if not os.path.exists(os.path.join(dest_test, class_dir)):
        os.mkdir(os.path.join(dest_test, class_dir))
    if not os.path.exists(os.path.join(dest_val, class_dir)):
        os.mkdir(os.path.join(dest_val, class_dir))

    for index, filename in enumerate(os.listdir(class_dir_path)):
            if index in val_indices:
                shutil.copy(os.path.join(class_dir_path, filename), os.path.join(dest_val, class_dir))
            elif index in test_indices:
                shutil.copy(os.path.join(class_dir_path, filename), os.path.join(dest_test, class_dir))
            else:
                shutil.copy(os.path.join(class_dir_path, filename), os.path.join(dest_train, class_dir))
