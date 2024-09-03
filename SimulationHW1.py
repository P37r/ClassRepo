import random
import stat
y = random.random()
z = 0
n = 0

def exp():
    z=0
    n=0
    while z<=1:
        z+= random.random()
        n+=1
    return n


a = []
b= []
c=[]
for i in range(100):
    a.append(exp())

for i in range(1000):
    b.append(exp())

for i in range(10000):
    c.append(exp())


print("a:", sum(a)/100)
print("b:", sum(b)/1000)
print("c:", sum(c)/10000)