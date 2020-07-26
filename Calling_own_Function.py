


try:
    p = open('C:\\Users\\amitk\\OneDrive\\Desktop\\amit1.xlsx')
    print(p)
except Exception:
    print("Problem with opening")
else:
    try:
        print(p.read())
    except Exception:
        print("Problem with reading")




