import pandas as pd
import time
import sys
from model_single import model_s

def main():

    for i in ['5']:#, '15', '30', '60']:
        eurusd = pd.read_csv('./data/EURUSD' + i + '.csv')
        ms = model_s(eurusd, i)
        ms.out()

    return 0


if __name__ == '__main__':
    i = time.time()
    main()
    f = time.time()
    # print(f - i)
