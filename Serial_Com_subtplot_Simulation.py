import serial
import matplotlib.pyplot as plt

s = serial.Serial('COM3', baudrate=9600, timeout=1)
interval = 0
Time = []
xAcce = []
yAcce = []
zAcce = []
gyroX=[]
gyroY=[]
gyroZ=[]
temmpe = []
timeSpan = 100

Oxa=.56
Oya=.1
Oza=0
Ogx=-1.0
Ogy=1.7
Ogz=-1


fig = plt.figure(figsize=(7, 70), dpi=100)





def searchh(m,n):
    temp = []
    swittemp = 0
    negtemp = 0
    nexttemp = 0
    for i in range(0, leng):
        if data[i] == m[0] and data[i + 1] == m[1] and data[i + 2] == m[2] and data[i + 3] == m[3] and data[
            i + 4] == ':' and data[i + 5] == ' ':
            if data[i + 6] == '-':
                swittemp = i + 7
                negtemp = 1
            else:
                swittemp = i + 6
            break
    for i in range(swittemp, leng):
        if data[i] == n:
            nexttemp = i
            break
    for i in range(swittemp, nexttemp):
        nom = (data[i])
        temp.append(nom)
    lentemp = len(temp)
    swiittemp = 0
    for i in range(0, lentemp):
        if temp[i] == '.':
            swiittemp = i + 1
            break
    temperature = 0.0
    for i in range(swiittemp, lentemp):
        temperature = int(temp[i]) / (10 ** (i - 2+1)) + temperature
    for i in range(0, swiittemp - 1):
        temperature = (10 ** i) * float(temp[swiittemp - 1 - i - 1]) + temperature
    if negtemp == 1:
        temperature = temperature * -1

    print(temperature)

    return temperature




Rerun=0
Oldtemp=0;
while True:
    try:
        data = s.readline().decode('ascii')
        leng = len(data)
        print(data[:])
        ################## Time sorting
        time = []
        for i in range(0, leng):
            if i > 5:
                if data[i] == " ":
                    break
                else:
                    nom = int(data[i])
                    time.append(nom)
                # print(data[i])
        timelen = len(time)
        tiime = 0
        for i in range(0, timelen):
            tiime = (10 ** i) * time[timelen - i - 1] + tiime
        print(tiime)

        ######### temp shorting
        temp = []
        swittemp = 0
        negtemp = 0
        nexttemp = 0
        for i in range(0, leng):
            if data[i] == 't' and data[i + 1] == 'e' and data[i + 2] == 'm' and data[i + 3] == 'p' and data[i + 4] == ':' and data[i + 5] == ' ':
                if data[i + 6] == '-':
                    swittemp = i + 7
                    negtemp = 1
                else:
                    swittemp = i + 6
                break
        for i in range(swittemp, leng):
            if data[i] == ' ':
                nexttemp = i
                break
        for i in range(swittemp, nexttemp):
            nom = (data[i])
            temp.append(nom)
        lentemp = len(temp)
        swiittemp = 0
        for i in range(0, lentemp):
            if temp[i] == '.':
                swiittemp = i + 1
                break
        temperature = 0.0
        for i in range(swiittemp, lentemp):
            temperature = int(temp[i]) / (10 ** (i - 2)) + temperature
        for i in range(0, swiittemp - 1):
            temperature = (10 ** i) * float(temp[swiittemp - 1 - i - 1]) + temperature
        if negtemp == 1:
            temperature = temperature * -1

        temperature=.95*Oldtemp+.05*temperature
        Oldtemp=temperature

        print(temperature)

        temperature =  temperature+.75
        AccelerationX = searchh('xAcc', ' ') - Oxa
        AccelerationY = searchh('yAcc', ' ') - Oya
        Accelerationz = searchh('ZAcc', '\r') - Oza
        gyroscopeX = searchh('gyrx', ' ') - Ogx
        gyroscopeY = searchh('gyry', ' ') - Ogy
        gyroscopeZ = searchh('gyrz', ' ') - Ogz


        def ajkk():
            print(321)


        def printt():
            print('ajllj')


        if interval > 2:
            Time.append(tiime)
            xAcce.append(AccelerationX)
            yAcce.append(AccelerationY)
            zAcce.append(Accelerationz)
            gyroX.append(gyroscopeX)
            gyroY.append(gyroscopeY)
            gyroZ.append(gyroscopeZ)
            temmpe.append(temperature)

            if len(Time) > timeSpan:
                timeplot = Time[(len(Time) - timeSpan):len(Time)]
                xAcceplot = xAcce[(len(Time) - timeSpan):len(Time)]
                yAcceplot = yAcce[(len(Time) - timeSpan):len(Time)]
                zAcceplot = zAcce[(len(Time) - timeSpan):len(Time)]
                gyroXplot = gyroX[(len(Time) - timeSpan):len(Time)]
                gyroYplot = gyroY[(len(Time) - timeSpan):len(Time)]
                gyroZplot = gyroZ[(len(Time) - timeSpan):len(Time)]
                temmpeplot = temmpe[(len(Time) - timeSpan):len(Time)]
            else:
                timeplot = Time
                xAcceplot = xAcce
                yAcceplot = yAcce
                zAcceplot = zAcce
                gyroXplot = gyroX
                gyroYplot = gyroY
                gyroZplot = gyroZ

                temmpeplot = temmpe
            Time = timeplot
            xAcce = xAcceplot
            yAcce = yAcceplot
            zAcce = zAcceplot
            gyroX = gyroXplot
            gyroY = gyroYplot
            gyroZ = gyroZplot
            temmpe = temmpeplot

            axtem = fig.add_subplot(311)
            axtem.plot(timeplot, temmpeplot)
            axtem.set_ylabel('Temperature')

            axGYRO = fig.add_subplot(312)
            axGYRO.plot(timeplot, gyroXplot, label='x')
            axGYRO.plot(timeplot, gyroYplot, label='y')
            axGYRO.plot(timeplot, gyroZplot, label='z')
            axGYRO.set_ylabel('Gyro Angular Velocity')
            plt.legend()

            axzAcc = fig.add_subplot(313)
            axzAcc.plot(timeplot, xAcceplot, label='x')
            axzAcc.plot(timeplot, yAcceplot, label='y')
            axzAcc.plot(timeplot, zAcceplot, label='z')
            axzAcc.set_ylabel('Acceleration')
            axzAcc.set_xlabel('Time')
            plt.legend()

            plt.show(block=False)
            plt.pause(.0001)
            fig.clf()

        interval = interval + 1
        Rerun = 0

    except Exception:
        Rerun = 1

    if Rerun==1:
        continue







