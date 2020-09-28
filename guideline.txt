Guideline

Environment settings:
Python (Tensorflow): 3.5.0
keras: 2.1.0 

The proposed single-agent DRL based F-AP selection approach is realized in this program, which intends to minimize long-term system energy consumption. 
Its main purpose is to solve the computation resource deployment problem that consists of F-AP selection sub-problem and F-APs' request forwarding sub-problem. 

FL_1.py, FL_2.py in the folder “training” is for the offline-training of DRL model. 
Start by running FL_1.py - the main program

FL_1.py, FL_2.py in the folder "online-decision making" is for the online F-AP selection. It's the same running procedure like offline-training, begin by running FL_1.py.
