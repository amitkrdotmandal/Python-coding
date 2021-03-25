import socket
socket.setdefaulttimeout(12000)
def alnnnl():
    c = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    c.connect(('10.194.109.223', 9999))
    ##password = input('Enter The password:')
    c.send(bytes(password, 'utf-8'))
    FirstMsg = c.recv(1024).decode()
    print(FirstMsg)
    if FirstMsg == 'Welcome':
        c.send(bytes('ok', 'utf-8'))
        MsgFrmAnUse = c.recv(1024).decode()
        print(MsgFrmAnUse)
        while True:
            Msg = input('Enter your msg:')
            c.send(bytes(Msg, 'utf-8'))
            if Msg == 'Break':
                break
    alnnnl()
c = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
c.connect(('10.194.109.223', 9999))
password = input('Enter The password:')
c.send(bytes(password, 'utf-8'))
FirstMsg = c.recv(1024).decode()
print(FirstMsg)
if FirstMsg == 'Welcome':
    c.send(bytes('ok', 'utf-8'))
    MsgFrmAnUse = c.recv(1024).decode()
    print(MsgFrmAnUse)
    while True:
        Msg = input('Enter your msg:')
        c.send(bytes(Msg, 'utf-8'))
        if Msg == 'Break':
            break
alnnnl()






