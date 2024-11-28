import qrcode
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from natsort import natsorted
import time
#iterators
line_number = 0
tray_number = 1 # this will go till 84
layer_number = 1 # can be either 1 or 2

#logic:
"""
1. as we iterate we are going to check which layer we are in and which tray
2. if either the layer or the tray has changed, add a row of empty spaces on the paper ## a row is 16 blocks long
"""
page_number=0
i=1
list_of_qr_codes = [i]
list_of_lists = []
### the csv is layed ou tlike this:
# "","bench","tray","layer","plots","treatments","reps"
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
with open('out_list.txt','r') as f:
	for lines in f:
		#input()
		#keep adding qr codes for as long as we are on the same rep.
		remove_return = lines.split('\n')[0]
		#print(remove_return)
		#print(lines.split('\n'))
		line_split = remove_return.split(',')
		#print(line_split)
		#input()
		
		# what it reads from file
		tray_read = line_split[0]
		layer_read = line_split[0]
		#print(layer_read)
		#print(layer_number)
		## logic bit
		if str(layer_number) == str(layer_read): #check layer
			#print('yash')
		## add qr to list to be added to sheets
			list_of_qr_codes.append(line_split)
		else:
			list_of_lists.append(list_of_qr_codes)
			#print(list_of_lists)
			# reset list
			list_of_qr_codes = []
			# add the latest read before continuing 
			i+=1
			list_of_qr_codes.append(i)
			list_of_qr_codes.append(line_split)
			
			layer_number = str(layer_read)
			


		
with open('test.txt', 'w+') as f:
	for line in list_of_lists:
		for lines in line:	
			f.write(str(lines)+'\n')





number_rep = 1 #how often do you want the same thing in the input file
### drawing a empty white box

columns = 24
rows = 16
number_of_Stickers_on_page = columns*rows



os.chdir(dir_path)
### removes any old made files from the working dir
for file in os.listdir(dir_path):
	if file.endswith('.bmp'):
		os.remove(file)
	elif file.endswith('.tiff'):
		os.remove(file)


### makes individual images that we later glue onto the pdf
#you may have to play with the sizes of the image and the font to get what you need
qr = qrcode.QRCode(
    version=10,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
plakken_timer = 1 ## if is 3 it will go next page
n=0
splitter_number =1 
font = ImageFont.truetype("arial.ttf", 40)
for j in list_of_lists:
	for i in j:
		#print(i)
		#input()
		if str(i) == str(splitter_number):
			splitter_number +=1
			length_string = len(str(i))
			img = Image.new('RGB', (50*length_string, 50*length_string), color = 'white')
			d = ImageDraw.Draw(img)
			d.text((0, 0), str(i), font=font,fill=(0,0,0))
			img.save(str(n)+".bmp")
			n+=1
		else:
			del i[0]	
			qr_text = ';'.join(i)
			#print(i)
			img = qrcode.make(qr_text)
			type(img)  # qrcode.image.pil.PilImage
			img.save(str(n)+".bmp")
			n+=1

		### defining an A4 page
width, height = 4961 , 7016
"""
max that would fit in the page:
width = 4961/236 = 21
height = 7016/236 = 29
"""
images= []
for file in natsorted(glob.glob("*.bmp")):
	images.append(file)
size =236,236
# to middle circle of sticker sheet
# top 16.75 mm = 396 Pixel
# side 19.75 mm = 467 Pixel
# between top to bot : 15.5 mm 366 Pixel
# between left to right 15.5 mm 366 Pixel
top=100
side=100
between=236
t=0
l=0
list_images=[]
for image in images:
	imagez = Image.open(image)
	imagez.thumbnail(size, Image.ANTIALIAS)
	list_images.append(imagez)
page = Image.new('RGB', (width, height), 'white')
n=0

number_of_stickers_done = 1
### pasting the previously made images onto the pdf
for pages in range(len(list_images)):
	if number_of_stickers_done < number_of_Stickers_on_page:
		for l in range(0, columns):
			for t in range(0, rows):
				if len(list_images) <= n:
					pass
				else:
					page.paste(list_images[n], box=(side + (t * between), top + (l * between)))
					n += 1
					number_of_stickers_done += 1
					#print(n)
	else:
		page.save((str(page_number)+'page.tiff'), interlace=False)
		page = Image.new('RGB', (width, height), 'white')
		number_of_stickers_done = 1
		page_number+=1
		#print(" saving page..." )
page.save((str(page_number+1)+'page.tiff'), interlace=False)
### removes any old made files from the working dir
for file in os.listdir(dir_path):
	if file.endswith('.bmp'):
		os.remove(file)
