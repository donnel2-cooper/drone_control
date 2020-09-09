def maximum(x,y):
    if x>y:
        return x
    elif x==y:
        return 'The numbers are equal'
    else:
        return y


### Checkcases ###
print(maximum(2,3)) #Gives 3
print(maximum(4,3)) #Gives 4
print(maximum(3,3)) #Gives 'The numbers are equal'