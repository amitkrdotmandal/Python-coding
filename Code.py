from numpy import *

element = matrix('1,2,3,4;4,3,5,6')
nodeCordinate = matrix('0, 0.0, 0; 1.0, 0, 0;1, 1, 0; 0, 1, 0; 1, 2, 0; 0, 2 .0 ')
doff = matrix('0, 0, 0; 0, 0, 0; 1, 1, 1; 1, 1, 1; 0, 1, 1; 0, 1, 1')

C = matrix('10000, 200, 0; 200, 10000, 0; 0, 0, 500')
thick = .2
maxLoad = -1000
maxTime = 10

RR = matrix([-1 / sqrt(3), 1 / sqrt(3)])
SS = RR
noEle, NPE = element.shape
noNode, DFPN = nodeCordinate.shape

k = 0
dof = zeros((DFPN * noNode))
for i in range(0, noNode):
    for j in range(0, 3):
        dof[k] = doff[i, j]
        k = k + 1
noDof = sum(doff)

# print(dof)


to = 0
tiiime = zeros((maxTime))
utglobal = zeros((DFPN * noNode, 1))
#utPlusDeltglobal = random.rand(DFPN * noNode, 1)
utPlusDeltglobal =matrix([[0], [0], [0], [0], [0], [0], [.870], [.4], [.20], [.230], [.980], [.10], [0], [.65], [.5], [0], [.65],
                  [.56]])

for t in range(0, maxTime, maxTime):  ####
    load = maxLoad * t / maxTime
    tiiime[to] = t
    to = to + 1
    rig = zeros((DFPN * noNode, 1))
    reg = matrix([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [load], [.5 * load], [0], [load],
                  [load]])  # here mey be problem
    ktg = zeros((DFPN * noNode, DFPN * noNode))

    utglobal = utPlusDeltglobal  # here may need to be modified
    ut1global = zeros(noNode)
    ut2global = zeros(noNode)
    ut3global = zeros(noNode)

    for i in range(0, noNode, 1):
        ut1global[i] = utglobal[3 * i, 0]
        ut2global[i] = utglobal[(3 * i) + 1, 0]
        ut3global[i] = utglobal[(3 * i) + 2, 0]

    for ele in range(0, noEle, 1):  #####
        # ele=1 ################################################
        KeLin = zeros((DFPN * NPE, DFPN * NPE))
        KeNonLin = zeros((DFPN * NPE, DFPN * NPE))
        Ke = zeros((DFPN * NPE, DFPN * NPE))
        ri = zeros((DFPN * NPE, 1))
        eleNode1 = element[ele, 0]
        eleNode2 = element[ele, 1]
        eleNode3 = element[ele, 2]
        eleNode4 = element[ele, 3]
        ng = matrix(
            [3 * eleNode1 - 3, 3 * eleNode1 - 2, 3 * eleNode1 - 1, 3 * eleNode2 - 3, 3 * eleNode2 - 2, 3 * eleNode2 - 1,
             3 * eleNode3 - 3,
             3 * eleNode3 - 2, 3 * eleNode3 - 1, 3 * eleNode4 - 3, 3 * eleNode4 - 2, 3 * eleNode4 - 1])

        X1 = nodeCordinate[eleNode1 - 1, 0]
        X2 = nodeCordinate[eleNode2 - 1, 0]
        X3 = nodeCordinate[eleNode3 - 1, 0]
        X4 = nodeCordinate[eleNode4 - 1, 0]
        Y1 = nodeCordinate[eleNode1 - 1, 1]
        Y2 = nodeCordinate[eleNode2 - 1, 1]
        Y3 = nodeCordinate[eleNode3 - 1, 1]
        Y4 = nodeCordinate[eleNode4 - 1, 1]
        XX = matrix([[X1], [X2], [X3], [X4]])
        YY = matrix([[Y1], [Y2], [Y3], [Y4]])

        U1t = ut1global[eleNode1 - 1]
        U2t = ut1global[eleNode2 - 1]
        U3t = ut1global[eleNode3 - 1]
        U4t = ut1global[eleNode4 - 1]
        V1t = ut2global[eleNode1 - 1]
        V2t = ut2global[eleNode2 - 1]
        V3t = ut2global[eleNode3 - 1]
        V4t = ut2global[eleNode4 - 1]
        W1t = ut3global[eleNode1 - 1]
        W2t = ut3global[eleNode2 - 1]
        W3t = ut3global[eleNode3 - 1]
        W4t = ut3global[eleNode4 - 1]
        UtVtWt = matrix([U1t, V1t, W1t, U2t, V2t, W2t, U3t, V3t, W3t, U4t, V4t, W4t])
        # print(UtVtWt)

        for m in range(0, 2, 1):
            for n in range(0, 2, 1):
                r = RR[0, n]
                s = SS[0, m]
                DNr = matrix([-(1 - s) / 4, (1 - s) / 4, (1 + s) / 4, -(1 + s) / 4])
                DNs = matrix([-(1 - r) / 4, -(1 + r) / 4, (1 + r) / 4, (1 - r) / 4])
                J11 = DNr.dot(XX)
                J12 = DNr.dot(YY)
                J21 = DNs.dot(XX)
                J22 = DNs.dot(YY)

                J = matrix([[J11[0, 0], J12[0, 0]], [J21[0, 0], J22[0, 0]]])
                detJ = linalg.det(J)
                Jstar = linalg.inv(J)

                Jmatrix = matrix([[Jstar[0, 0], Jstar[0, 1], 0, 0, 0, 0],
                                  [0, 0, Jstar[0, 0], Jstar[0, 1], 0, 0],
                                  [0, 0, 0, 0, Jstar[0, 0], Jstar[0, 1]],
                                  [Jstar[1, 0], Jstar[1, 1], 0, 0, 0, 0],
                                  [0, 0, Jstar[1, 0], Jstar[1, 1], 0, 0],
                                  [0, 0, 0, 0, Jstar[1, 0], Jstar[1, 1]]])

                srMatrix = matrix([[DNr[0, 0], 0, 0, DNr[0, 1], 0, 0, DNr[0, 2], 0, 0, DNr[0, 3], 0, 0],
                                   [DNs[0, 0], 0, 0, DNs[0, 1], 0, 0, DNs[0, 2], 0, 0, DNs[0, 3], 0, 0],
                                   [0, DNr[0, 0], 0, 0, DNr[0, 1], 0, 0, DNr[0, 2], 0, 0, DNr[0, 3], 0],
                                   [0, DNs[0, 0], 0, 0, DNs[0, 1], 0, 0, DNs[0, 2], 0, 0, DNs[0, 3], 0],
                                   [0, 0, DNr[0, 0], 0, 0, DNr[0, 1], 0, 0, DNr[0, 2], 0, 0, DNr[0, 3]],
                                   [0, 0, DNs[0, 0], 0, 0, DNs[0, 1], 0, 0, DNs[0, 2], 0, 0, DNs[0, 3]]])
                UtVtWttranspose=UtVtWt.transpose()
                srMatrixUtVtWttranspose =srMatrix.dot(UtVtWttranspose)


                dUdXdVdXdWdXdUdYdVdYdWdY = Jmatrix.dot(srMatrixUtVtWttranspose)

                bma = matrix([[dUdXdVdXdWdXdUdYdVdYdWdY[0, 0] + 1, dUdXdVdXdWdXdUdYdVdYdWdY[1, 0],
                               dUdXdVdXdWdXdUdYdVdYdWdY[2, 0], 0, 0, 0],
                              [0, 0, 0, dUdXdVdXdWdXdUdYdVdYdWdY[3, 0], dUdXdVdXdWdXdUdYdVdYdWdY[4, 0] + 1,
                               dUdXdVdXdWdXdUdYdVdYdWdY[5, 0]],
                              [dUdXdVdXdWdXdUdYdVdYdWdY[3, 0], dUdXdVdXdWdXdUdYdVdYdWdY[4, 0] + 1,
                               dUdXdVdXdWdXdUdYdVdYdWdY[5, 0], dUdXdVdXdWdXdUdYdVdYdWdY[0, 0] + 1,
                               dUdXdVdXdWdXdUdYdVdYdWdY[1, 0], dUdXdVdXdWdXdUdYdVdYdWdY[2, 0]]])
                JmatrixsrMatrix =Jmatrix.dot(srMatrix)
                BMatLin = bma.dot(JmatrixsrMatrix)
                CBMatLin = C.dot(BMatLin)
                BMatLintranspose = BMatLin.transpose()

                KeLin = KeLin + ((BMatLintranspose.dot(CBMatLin)) * detJ * thick)

                epsXXt = dUdXdVdXdWdXdUdYdVdYdWdY[0, 0] + 1 / 2 * pow(dUdXdVdXdWdXdUdYdVdYdWdY[0, 0], 2) + 1 / 2 * pow(
                    dUdXdVdXdWdXdUdYdVdYdWdY[1, 0], 2) + 1 / 2 * pow(dUdXdVdXdWdXdUdYdVdYdWdY[2, 0], 2)
                epsYYt = dUdXdVdXdWdXdUdYdVdYdWdY[4, 0] + 1 / 2 * pow(dUdXdVdXdWdXdUdYdVdYdWdY[3, 0], 2) + 1 / 2 * pow(
                    dUdXdVdXdWdXdUdYdVdYdWdY[4, 0], 2) + 1 / 2 * pow(dUdXdVdXdWdXdUdYdVdYdWdY[5, 0], 2)
                epsXYt = 1 / 2 * (dUdXdVdXdWdXdUdYdVdYdWdY[1, 0] + dUdXdVdXdWdXdUdYdVdYdWdY[3, 0] +
                                  dUdXdVdXdWdXdUdYdVdYdWdY[0, 0] * dUdXdVdXdWdXdUdYdVdYdWdY[3, 0] +
                                  dUdXdVdXdWdXdUdYdVdYdWdY[1, 0] * dUdXdVdXdWdXdUdYdVdYdWdY[4, 0] +
                                  dUdXdVdXdWdXdUdYdVdYdWdY[2, 0] * dUdXdVdXdWdXdUdYdVdYdWdY[5, 0])
                epsmatrx= matrix([[epsXXt], [epsYYt], [2 * epsXYt]])

                StXXStYYStXY = C.dot(epsmatrx)

                BMatNonLin = Jmatrix.dot(srMatrix)

                CnonLin = matrix([[StXXStYYStXY[0, 0], 0, 0, StXXStYYStXY[1, 0], 0, 0],
                                  [0, StXXStYYStXY[0, 0], 0, 0, StXXStYYStXY[1, 0], 0],
                                  [0, 0, StXXStYYStXY[0, 0], 0, 0, StXXStYYStXY[1, 0]],
                                  [StXXStYYStXY[1, 0], 0, 0, StXXStYYStXY[2, 0], 0, 0],
                                  [0, StXXStYYStXY[1, 0], 0, 0, StXXStYYStXY[2, 0], 0],
                                  [0, 0, StXXStYYStXY[1, 0], 0, 0, StXXStYYStXY[2, 0]]])

                BMatNonLintranspose = BMatNonLin.transpose()
                CnonLinBMatNonLin = CnonLin.dot(BMatNonLin)

                KeNonLin = KeNonLin + ((BMatNonLintranspose.dot(CnonLinBMatNonLin)) * detJ * thick)
                ri = ri +BMatLintranspose.dot(StXXStYYStXY) * detJ * thick

        Ke = KeLin + KeNonLin

        # k=2
        # print(rig[ng[0, k], 0])
        # print(Ke)

        for k in range(0, DFPN * NPE, 1):
            rig[ng[0, k], 0] = rig[ng[0, k], 0] + ri[k, 0]
            for l in range(0, DFPN * NPE, 1):
                ktg[ng[0, k], ng[0, l]] = ktg[ng[0, k], ng[0, l]] + Ke[k, l]

    rigredu = zeros((noDof, 1))
    regredu = zeros((noDof, 1))
    reduktg = zeros((noDof, noDof))
    idofi = 0
    for i in range(0, DFPN * noNode, 1):
        if dof[i] == 1:
            rigredu[idofi, 0] = rig[i, 0]
            regredu[idofi, 0] = reg[i, 0]
            idofj = 0
            for j in range(0, DFPN * noNode, 1):
                if dof[j] == 1:
                    reduktg[idofi, idofj] = ktg[i, j]
                    idofj = idofj + 1

            idofi = idofi + 1

    deltUglobalRedu = (linalg.inv(reduktg)).dot((regredu - rigredu))
    deltUglobal = zeros((DFPN * noNode, 1))
    idofi = 0
    for i in range(0, DFPN * noNode, 1):
        if dof[i] == 1:
            deltUglobal[i, 0] = deltUglobalRedu[idofi, 0]
            idofi = idofi+1

    utPlusDeltglobal = utglobal+deltUglobal
    print(deltUglobal)






    










   # print(utglobal)
    #print(deltUglobal)
    #print(utPlusDeltglobal)









































































