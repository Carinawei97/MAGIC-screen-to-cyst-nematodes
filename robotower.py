""" install: For example codes using the pi relay: git clone https://github.com/johnwargo/seeed-studio-relay-board

requires these files in the folder to run:
robotower.py
relay_lib_seeed.py

to run; python robotower.py
"""
'''
List of machine names:

Baymax 
Lilo 
Goofy 
Jasmine 
Sebastian
EVE 

'''

machine_name = 'Goofy'

import RPi.GPIO as GPIO
import time
import os
import subprocess
import os.path
import glob
from relay_lib_seeed import *
import datetime

#0. checks if we ran out of plates
def empty_stack_check():
	global machine_status
	prox_sensor_val = GPIO.input(prox_sensor_empty)
	if prox_sensor_val == 1:
		relay_off(motor)
		machine_status = 'not_running'
		return
	else:
		#relay_on(motor)
		#wait_sequence()
		pass

#1. keeps the motor running until it has seen a hall sensor for x amount of uninterupted reads. turns off relay, then takes an image
def roll_sequence():
	global state_list_check
	global state_strictness_treshold
	global state
	relay_on(motor)
	while state == 'roll':
		current_state = GPIO.input(sensor_signal)
		if len(state_list_check) > state_strictness_treshold and list(set(state_list_check))[0] == 0: # checks if all states are of the same value and go over a treshold, if so, take image.
			relay_off(motor)
			take_image()
		else:
			if len(set(state_list_check)) > 1: # checks if the states are all the same, if not: reinitiate the list
				state_list_check = []
			else:
				state_list_check.append(current_state) # adds current state of hall sensor to list

#2. is called by the roll sequence when the hall sensor is triggered, and the relay is turned off. this function takes an image and then checks if that image is taken using image_check(), if so it restarts the relay on and goes to wait_sequence()
def take_image():
	global command
	global state_list_check
	global check
	global state
	state_list_check = []
	state = 'wait'
	relay_on(lights)
	check = 1
	time.sleep(0.05)
	subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	image_check()
	relay_off(lights)
	relay_on(motor)


#3. checks if the image files is written (this is out.bmp) and then renames it to n.relay_on(motor)
def image_check():
	global check
	global n
	global machine_name
	time_now = datetime.datetime.now()
	time_string = time_now.strftime("%d-%b-%Y_%H_%M_%S.%f")
	while check == 1:
		if  os.path.isfile('out.bmp'): 
			os.rename('out.bmp', (time_string+'_'+machine_name+'.bmp'))	
			n+=1
			check = 0
		else:
			pass

# 4. this is right after taking the image, we have started the motor again, but the hall sensor is still triggered, so we want to ignore that for x amount of reads. Works on the same principle as roll_sequence()
def wait_sequence():
	global state_list_check
	global state_strictness_treshold
	global state
	while state == 'wait':
		current_state = GPIO.input(sensor_signal)
		if len(state_list_check) > state_strictness_treshold and list(set(state_list_check))[0] == 1:
			state_list_check = []
			state = 'roll'
		else:
			state_list_check.append(current_state)
			if len(set(state_list_check)) > 1:
				state_list_check = []
			else:
				pass

#5. checks if the machine is full on restacking end
def full_stack_check():
	global machine_status
	prox_sensor_val = GPIO.input(prox_sensor_full)
	if prox_sensor_val == 0:
		relay_off(motor)
		machine_status = 'not_running'
		return
	else:
		pass

#6. checks if button is pressed
def button_check():
	global machine_status
	global button_sensor
	button_val = GPIO.input(button_sensor)
	if button_val == 0:
		button(machine_status)
	else:
		pass

#7. changes status dependent on current status
def button(status):
	global machine_status
	if status == 'running':
		machine_status = 'not_running'
		time.sleep(0.2) # not sure if needed anymore
	else:
		machine_status = 'running'
		time.sleep(0.2)# not sure if needed anymore
		#relay_on(motor)

#defining relays
motor = 4
lights = 3
shorting_motor = 2

#setting up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#setting work dir and imaging commands.
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
command = r'libcamera-still -t 500 --awb fluorescent -e bmp -o out.bmp' # add -n to not have preview window
n=len(glob.glob1(dir_path,"*.bmp"))
state_strictness_treshold = 3 #treshold for stopping using hall sensor
 
#sensors, the pin numbers are BCM and are as labled on the relay board
sensor_signal = 23 #this is the hall sensor
prox_sensor_full = 18
prox_sensor_empty = 17
button_sensor = 27

#initiating the sensors
GPIO.setup(sensor_signal, GPIO.IN)
GPIO.setup(prox_sensor_full, GPIO.IN)
GPIO.setup(prox_sensor_empty, GPIO.IN)
GPIO.setup(button_sensor, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#parameters and lists
state_list_check = []
wait_time=2
state = 'roll'
button_val = GPIO.input(button_sensor)
machine_status = 'running'
relay_on(motor)

while True:
	empty_stack_check()
	if machine_status == 'running':
		roll_sequence()
		wait_sequence()
		full_stack_check()
	else:	
		button_check()
