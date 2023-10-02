# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 00:07:01 2023

@author: vivek chandra
"""
import random
import time
random.seed(time.time())

class_lst_string = """Adhikary Tirna,
             Mohammad Bayat,
             Atharva Manoj Chiplunkar,
             Souvik Dey,
             Jack Doyle,
             Quynh Duong,
             Kevin George,
             Ben Giacalone,
             Pradeep Kumar Gontla,
             Xinyu Hu,
             Ashwin Sharad Kherde,
             Maitreya Kocharekar,
             Gerrit Krot, 
             Huawei Lin,
             Emily Liu, 
             Khushi Mahesh, 
             Marie Mellor, 
             Anikhet Mulky, 
             Kruthi Nagabhushan, 
             Calvin Nau, 
             Anh Nguyen, 
             Bao Nguyen, 
             Jiakai Peng, 
             Snehil Sharma, 
             Kaustubh Narendranath Shetty, 
             Sicy Shi, 
             Athina Stewart,
             Alejandro Yaber Llanos"""
class_lst = []

for names in class_lst_string.split(","):
    class_lst.append(names.strip())

class_set = set(class_lst)
class_dict = {}
    
for i in range(0,len(class_lst),3):
    grp_num = "Group:"+str((i//3)+1)
    grp_lst = random.sample(class_set,3)
    
    class_dict[grp_num] = grp_lst
    
    
    for student in grp_lst:
        class_set.remove(student)
    
    if len(class_set)<3:
        break

for element in class_set:
    new_grp = random.sample(class_dict.keys(),1)
    
    class_dict[new_grp[0]].append(element)

for grp in class_dict.items():
    print(grp[0]," -> ",(",").join(grp[1]))
    
    

