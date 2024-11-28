import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

number_of_reps = 50

with open('in_list.txt','r') as f:
    with open('out_list.txt', 'w') as ff:
        for i in f:
            line = i.split('\n')[0]
            for j in range(0,number_of_reps):
                ff.write(line +";"+str(j)+'\n')
            
