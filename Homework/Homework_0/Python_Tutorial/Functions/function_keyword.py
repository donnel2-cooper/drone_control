def func(a,b=5,c=10):
    print('a is',a,'and b is',b,'and c is',c)
    
func(3,7) #Expect 3 7 10
func(25,c=24) #Expect 25 5 24
func(c=50,a=100) #Expect 100 5 50