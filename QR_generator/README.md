# OS

Tested on Windwos 11. Arial font was included in Windows, this may be changed as needed.

# Software

numpy         2.2.6

opencv-python 4.12.0.88

pillow        11.3.0

pyzbar        0.1.9


# Testing of script

To test the script download the example file `in_list.txt`. 

## Generate names

To generate list of names needed for qr codes run `python3 make_names.py`.

## Generate qr codes

**Note:** this script makes many small files, this may cause issues on a remotely mounted storage. 

Generate qr codes on a A4 layout using; `python3 generate.py`.



