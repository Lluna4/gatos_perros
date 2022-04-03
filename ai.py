import sklearn
import numpy as np
import pandas as pd
import os
import csv

for nombre in os.listdir(r"dogs-vs-cats\train\train"):
    with open("n.csv", "a") as f:
        writer = csv.writer(f)
        if "cat" in nombre:
            nombre = 0
        else:
            nombre = 1
        writer.writerow([nombre, ])
