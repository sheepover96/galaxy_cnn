
import pandas as pd
import csv
import os

if __name__ == '__main__':
    dropout_data = pd.read_csv('dataset/dropout1.csv', header=None)
    paths = dropout_data[2:4]
    new_row1 = []
    new_row2 = []
    with open('dataset/dropout_4c_multi_class.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for idx, data in dropout_data.iterrows():
            img_id = str( data[0] )
            print(img_id)
            img_no = str( data[1] )
            label = str( data[5] )
            first_path = data[2]
            fits_name = first_path[-6:]
            fits_mid_path = first_path[:-6]
            fits_paths = data[2:5]
            if fits_name.count('1.fits') > 0:
                pass
                fits_paths = [fits_path for fits_path in fits_paths]
                fits_paths.append(os.path.join(fits_mid_path, '4.fits'))
            else:
                if label == "1":
                    label = str(2)
                tmp_fits_paths = fits_paths
                fits_paths = [os.path.join( fits_mid_path, '1.fits' )]
                fits_paths.extend( [ fits_path for fits_path in tmp_fits_paths ]  )

            new_data_row = [img_id, img_no]
            new_data_row.extend(fits_paths)
            new_data_row.append(label)

            writer.writerow(new_data_row)

    #for path in paths:
    #    print(path)
    #    first_path = path
    #    part = path[:-6]
    #    tail = path[-6:]
    #    if tail.count('1.fits') > 0:
    #        new_row1.append(part + '4.fits')
    #    else:
    #        new_row1.append(part + '1.fits')
    #    new_row2.append(part + '5.fits')


    #last_row = dropout_data[dropout_data.shape[1] - 1]
    #dropout_data[dropout_data.shape[1] - 1] = new_row1
    #dropout_data[dropout_data.shape[1]] = new_row2
    #dropout_data[dropout_data.shape[1] + 1] = last_row
    #dropout_data.to_csv('dataset/dropout3.csv', index=None, header=None)
