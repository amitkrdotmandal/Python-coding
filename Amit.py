
#This is for learning python
"""
this is my first
code
in python
as Matlab isn't working offline
"""
# first part
"""
print("Balnkjnd",end="")
print("this is Amit")
var1="give me some sunshine"
var2=12
var3=16

var4=var2+var3
var5="13"
print(type(var4))
print(int(var5))
"""
"""
float()
str()
"""


"""
print("Enter your first number")
n1=input()

print("Enter your second number")
n2=input()

print("Your result is given below after adding")
print(int(n1)+int(n2))
"""
#String
"""
Myname="Amit Kumar Mandal"
print(Myname[0:4])
print(len(Myname))
"""

#list
"""
liist=["amit", "kumar", "mandal",1,12]
print(liist[0])
liistt=[1,2,3,5]
liistt.append(60)
liistt[1]=22

print(liistt)
"""
#dictionary
"""
khana={"Amit":"machh","Tana":"egg","maa":"ruti","baba":{"ratri":"muri","dupur":"bhat"}}
print(khana)
print(khana["baba"]["ratri"])
"""
#if elif and else
"""
a=11
b=31
c=b%a
if a>5:
    print("a is greater than 5")
    if b>30:
        print("b is greater than 30")
        if c==0:
             print("their reminder is zero")
        else:
            print("their reminder is nonzero")
    elif b<25:
        print("b is less than 25")
        if c==0:
             print("their reminder is zero")
        else:
            print("their reminder is nonzero")
    else:
        print("b is neither greater than 30 nor less than 25")
        if c==0:
            print("their reminder is zero")
        else:
            print("their reminder is nonzero")

else:
    print("a is less than or equal to 5")
    if b>30:
        print("b is greater than 30")
        if c==0:
            print("their reminder is zero")
        else:
            print("their reminder is nonzero")
    elif b<25:
        print("b is less than 25")
        if c==0:
            print("their reminder is zero")
        else:
            print("their reminder is nonzero")
    else:
        print("b is neither greater than 30 nor less than 25")
        if c==0:
             print("their reminder is zero")
        else:
            print("their reminder is nonzero")

"""

##loop
#while loop
"""
i=1
while i<10:
    print("no. of iteration "+str(i))
    i=i+1
"""
#for loop


"""
for i in range(1,20,2):
    if i>12:
        break
    print("no. of iteration " + str(i))
print("Sry you are greater than 12" )
"""

#array
"""
from array import *
vals=array('f',[1.2,10,44])
vals[1]=101


val=vals[1]+vals[2]

print(vals)
"""


#function

def greet():
    print("define it")


def addd_sub(a,b):
    c=a+b
    d=a-b
    return(c,d)

greet()
import math as m
a,b=(addd_sub(2,5))
d=m.sqrt(36)
print(d)
"""
"""
#array
"""
from array import *
print("Enter no. that you want to enter:")
ar=array('i',[])
n=int(input())
for i in range(1,n,1):
    ar.append(int(input()))

print(ar)

print("Enter the value to search")
b=int(input())
for i in range(0,len(ar),1):
    if ar[i]==b:
        print(i)
        break
"""

