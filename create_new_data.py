
import pandas as pd

if __name__ == '__main__':
    dropout_data = pd.read_csv('dataset/dropout1.csv', header=None)
    paths = dropout_data[2]
    new_row1 = []
    new_row2 = []
    for path in paths:
        part = path[:-6]
        new_row1.append(part + '4.fits')
        new_row2.append(part + '5.fits')


    last_row = dropout_data[dropout_data.shape[1] - 1]
    dropout_data[dropout_data.shape[1] - 1] = new_row1
    dropout_data[dropout_data.shape[1]] = new_row2
    dropout_data[dropout_data.shape[1] + 1] = last_row
    dropout_data.to_csv('dataset/dropout3.csv', index=None, header=None)

