def print_max(a,b):
    if a > b:
        print(a, 'is maximum')
    elif a < b:
        print(b,'is maximum')
    else:
        print(a, 'is equal to', b)
        
### Checkcases ###

# Directly pass in values
print_max(3, 4)

# With Variables
x = 5
y = 7
print_max(x, y)

#Equal Values
print_max(1, 1)