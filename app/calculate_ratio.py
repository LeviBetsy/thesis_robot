
result = [(200.0, 20), (145.0, 30), (114.25, 40)]

#ratio returned will be centimeter/pixel. 
if __name__ == "__main__":
    ratio = 0
    for p, dist in result:
        # ratio += float(dist)/p
        f = (p*dist)/11
        print(f)
        ratio += f

    avg = ratio/3.0
    print(avg) #0.219 first run