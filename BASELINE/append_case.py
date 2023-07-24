
import sys
import pandas as pd
import random
import pickle
import numpy as np
import os
import time


if __name__ == "__main__":
    with open("train_data.pkl","rb") as f:
        stock_pct = pickle.load(f)
    num_iter = int(sys.argv[1])
    cases = {}
    for _ in range(num_iter):
        random_stock = random.sample(list(stock_pct.keys()),400)
        ret = []
        for _i in range(200):
            ret.append(stock_pct[random_stock[_i]].to_list())
        for _i in range(200,400):
            ret.append((stock_pct[random_stock[_i]] * -1).to_list())
        ret = np.array(ret)
        case_name = "/".join(random_stock)
        cases[case_name] = {
            "mean":ret.sum(axis=0).mean(),
            "std":ret.sum(axis=0).std()
        }
    with open("cases_"+sys.argv[2]+".pkl","wb") as f:
        pickle.dump(cases,f)


    