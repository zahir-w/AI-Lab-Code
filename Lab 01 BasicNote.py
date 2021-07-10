# Question 01
print("%s" % "hello, world")
print("{}".format("hello, world"))


# Question 02
print(4*5)
print(4**5)
print(4/2)
print(4//2)
print((2+2)*3)
print(2+(2*3))
print(2+2*3)

a = 10; b = 20; c = "15"
print(a == b)
print(a > b)
print(a < b)
print(a != b)
print(a == c)
print(a >= c)


# Question 03
mylist = [1, "3", 6, -5, "-4", 2]
def orig(a):
    return int(a)
mylist.sort(key = orig)
print(mylist)

def orig_squa(a):
    return int(a)**2
mylist.sort(key = orig_squa)
print(mylist)


# Question 04
mylist = [1, 2, 2, 3, 3, 4, 5, 5]
tempset = set(mylist)
mylist = list(tempset)
print(mylist)


# Question 05
dict1 = {"one":1,"two":2}
dict2 = {1:"one",2:"two"}
del(dict1["one"])
print(dict1)
dict1["three"] = 3
print(dict1)
for k in dict2.keys():
    dict1[k] = dict2[k]
print(dict1)
if 3 in dict2.keys():
    del(dict2[3])
else:
    print("3 isn't the key of the dict2")


# Question 06
mylist = [2, 4, 11., 6, 5, -4, 14, 9]
print(max(mylist))
print(min(mylist))
print(sum(mylist))


# Question 07
mydict = {"one":1, "two":2, "three":3}
for key,value in mydict.items():
    print(key, ":", value)


# Question 08
mydict = {"one":1, "two":2, "three":3}
for key,value in enumerate(mydict):
    print(key,value)


# Question 09
mystr = "hello, python and world!"
print(mystr[3])
print(mystr[-4:])
print(mystr.replace("o","k"))
print(mystr.find("n"))
print(mystr.find("and"))


# Question 10
def cal_sum(ls):
    print(sum(ls))
mylist = [1, 2, 3, 4, 5]
cal_sum(mylist)

class school():
    def sch(self,b):
        print("The student is from %s." % b)
class campus():
    def cam(self,b):
        print("The student is from %s." % b)
class department():
    def dep(self,b):
        print("The student is from %s." % b)
class student(school,campus,department):
    pass
student1 = student()
student1.sch("SCUT")
student1.cam("GZIC")
student1.dep("WUSIE")


# Question 11
import numpy as np
arr = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
print(arr.max())
print(arr.min())
print(arr.mean())
print(np.median(arr))
print(arr.var())
print(arr.std())
print(arr.max(axis = 0))
print(arr.max(axis = 1))
print(arr.min(axis = 0))
print(arr.min(axis = 1))
print(arr.mean(axis = 0))
print(arr.mean(axis = 1))
print(np.median(arr,axis = 0))
print(np.median(arr,axis = 1))
print(arr.var(axis = 0))
print(arr.var(axis = 1))
print(arr.std(axis = 0))
print(arr.std(axis = 1))


# Question 12
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.sin(x)
plt.plot(x, y)
plt.show()
