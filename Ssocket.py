import socket
BigMsg = 'kk'
s = socket. socket()
print("socket created")
hostName='10.194.109.223'
#hostName=socket.gethostname()
print(hostName)
IPADD=socket.gethostbyname(hostName)
print(IPADD)
s.bind((IPADD, 9999))
s.listen(6)
print('waiting for connection...')

while True:
    c, addrs = s.accept()
    password = c.recv(1024).decode()
    print('connected with ', addrs, password)
    if password=="Sas":
        c.send(bytes('Welcome','utf-8'))
        ccc=c.recv(1024).decode()
        if ccc=='ok':
            c.send(bytes(BigMsg, 'utf-8'))


        BigMsg = ''
        while True:
            msg= c.recv(1024).decode()
            if msg == 'Break':
                break
            print(msg)
            BigMsg=BigMsg + msg


    else:
        c.send(bytes('Not Allowed', 'utf-8'))
    c.close()




