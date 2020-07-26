import xlrd
from numpy import *
import xlsxwriter
import random

outworkbook = xlsxwriter.Workbook("C:\\Users\\amitk\\OneDrive\\Desktop\\amit1.xlsx")
outsheet = outworkbook.add_worksheet()
outsheet2 = outworkbook.add_worksheet()

pth ="C:\\Users\\amitk\\OneDrive\\Desktop\\F1.xlsx"
workbookk1=xlrd.open_workbook(pth)


worksheet_element=workbookk1.sheet_by_index(0)
rows_ele=worksheet_element.nrows
columns_ele=worksheet_element.ncols
element = ones((rows_ele, columns_ele), dtype = int)
for i in range(0, rows_ele):
    for j in range(0, columns_ele):
        element[i, j] = int(worksheet_element.cell_value(i, j))
#print(element)

worksheet_node=workbookk1.sheet_by_index(1)
rows_node=worksheet_node.nrows
columns_node=worksheet_node.ncols
nodeCordinate= ones((rows_node, columns_node))
for i in range(0, rows_node):
    for j in range(0, columns_node):
        nodeCordinate[i, j] = worksheet_node.cell_value(i, j)
#print(nodeCordinate)

worksheet_dof=workbookk1.sheet_by_index(2)
rows_dof=worksheet_dof.nrows
columns_dof=worksheet_dof.ncols
doff= ones((rows_dof, columns_dof), dtype = int)
for i in range(0, rows_dof):
    for j in range(0, columns_dof):
        doff[i, j] = worksheet_dof.cell_value(i, j)
#print(doff)

worksheet_force=workbookk1.sheet_by_index(3)
rows_force=worksheet_force.nrows
columns_force=worksheet_force.ncols
forcee= ones((rows_force, columns_force))
for i in range(0, rows_force):
    for j in range(0, columns_force):
        forcee[i, j] = worksheet_force.cell_value(i, j)
#print(forcee)








Cc = matrix([[10000, 2000, 0], [2000, 10000, 0], [0, 0, 5000]])          ###############################################################################
C=Cc*1
thick = .2
maxLoad = -20

maxTime = 200

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

k = 0
forceei = zeros((DFPN * noNode,1))
for i in range(0, noNode):
    for j in range(0, 3):
        forceei[k,0] = forcee[i, j]
        k = k + 1

#print(forceei)




to = 0
tiiime = zeros((maxTime))
utglobal = zeros((DFPN * noNode, 1))
utPlusDeltglobal = zeros((DFPN * noNode, 1))

#print(utPlusDeltglobal)
for i in range(0, DFPN * noNode, 1):
    if dof[i] == 1:
        utPlusDeltglobal[i,0]=random.uniform(0, .1)
    else:
        utPlusDeltglobal[i, 0]=0

#print(utPlusDeltglobal)



for t in range(0, 10, 1):  ####
    load = (maxLoad * t / maxTime)-.1
    tiiime[to] = t
    to = to + 1
    rig = zeros((DFPN * noNode, 1))
    #reg = matrix([[0], [0], [0], [0], [0], [0], [0], [0], [-load], [0], [0], [-load], [0], [0], [0], [.0], [0],[0]])  # here mey be problem
    reg= forceei*load
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
                UtVtWttranspose = UtVtWt.transpose()
                srMatrixUtVtWttranspose = srMatrix.dot(UtVtWttranspose)

                dUdXdVdXdWdXdUdYdVdYdWdY = Jmatrix.dot(srMatrixUtVtWttranspose)

                bma = matrix([[dUdXdVdXdWdXdUdYdVdYdWdY[0, 0] + 1, dUdXdVdXdWdXdUdYdVdYdWdY[1, 0],
                               dUdXdVdXdWdXdUdYdVdYdWdY[2, 0], 0, 0, 0],
                              [0, 0, 0, dUdXdVdXdWdXdUdYdVdYdWdY[3, 0], dUdXdVdXdWdXdUdYdVdYdWdY[4, 0] + 1,
                               dUdXdVdXdWdXdUdYdVdYdWdY[5, 0]],
                              [dUdXdVdXdWdXdUdYdVdYdWdY[3, 0], dUdXdVdXdWdXdUdYdVdYdWdY[4, 0] + 1,
                               dUdXdVdXdWdXdUdYdVdYdWdY[5, 0], dUdXdVdXdWdXdUdYdVdYdWdY[0, 0] + 1,
                               dUdXdVdXdWdXdUdYdVdYdWdY[1, 0], dUdXdVdXdWdXdUdYdVdYdWdY[2, 0]]])
                JmatrixsrMatrix = Jmatrix.dot(srMatrix)
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
                epsmatrx = matrix([[epsXXt], [epsYYt], [2 * epsXYt]])

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
                ri = ri + BMatLintranspose.dot(StXXStYYStXY) * detJ * thick

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

    sspom=regredu - rigredu
    ssspom = linalg.inv(reduktg)
    deltUglobalRedu = ssspom.dot(sspom)
    #print(reduktg)
    #print(sspom)

    deltUglobal = zeros((DFPN * noNode, 1))
    idofi = 0
    for i in range(0, DFPN * noNode, 1):
        if dof[i] == 1:
            deltUglobal[i, 0] = deltUglobalRedu[idofi, 0]
            idofi = idofi + 1

    utPlusDeltglobal = utglobal + deltUglobal
    #print(utPlusDeltglobal)
    iterationParameter = 0
    iterationNo = 0
    while iterationParameter == 0:
        iterationNo = iterationNo + 1
        rigtPlusDelt = zeros((DFPN * noNode, 1))
        ktgtPlusDelt = zeros((DFPN * noNode, DFPN * noNode))

        utPlusDelt1global = zeros(noNode)
        utPlusDelt2global = zeros(noNode)
        utPlusDelt3global = zeros(noNode)

        for i in range(0, noNode, 1):
            utPlusDelt1global[i] = utPlusDeltglobal[3 * i, 0]
            utPlusDelt2global[i] = utPlusDeltglobal[(3 * i) + 1, 0]
            utPlusDelt3global[i] = utPlusDeltglobal[(3 * i) + 2, 0]

        for ele in range(0, noEle, 1):
            KeLintPlusDelt = zeros((DFPN * NPE, DFPN * NPE))
            KeNonLintPlusDelt = zeros((DFPN * NPE, DFPN * NPE))
            KetPlusDelt = zeros((DFPN * NPE, DFPN * NPE))
            ritPlusDelt = zeros((DFPN * NPE, 1))
            eleNode1 = element[ele, 0]
            eleNode2 = element[ele, 1]
            eleNode3 = element[ele, 2]
            eleNode4 = element[ele, 3]
            ng = matrix(
                [3 * eleNode1 - 3, 3 * eleNode1 - 2, 3 * eleNode1 - 1, 3 * eleNode2 - 3, 3 * eleNode2 - 2,
                 3 * eleNode2 - 1, 3 * eleNode3 - 3,
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

            U1tPlusDelt = utPlusDelt1global[eleNode1 - 1]
            U2tPlusDelt = utPlusDelt1global[eleNode2 - 1]
            U3tPlusDelt = utPlusDelt1global[eleNode3 - 1]
            U4tPlusDelt = utPlusDelt1global[eleNode4 - 1]
            V1tPlusDelt = utPlusDelt2global[eleNode1 - 1]
            V2tPlusDelt = utPlusDelt2global[eleNode2 - 1]
            V3tPlusDelt = utPlusDelt2global[eleNode3 - 1]
            V4tPlusDelt = utPlusDelt2global[eleNode4 - 1]
            W1tPlusDelt = utPlusDelt3global[eleNode1 - 1]
            W2tPlusDelt = utPlusDelt3global[eleNode2 - 1]
            W3tPlusDelt = utPlusDelt3global[eleNode3 - 1]
            W4tPlusDelt = utPlusDelt3global[eleNode4 - 1]
            UtVtWttPlusDelt = matrix(
                [U1tPlusDelt, V1tPlusDelt, W1tPlusDelt, U2tPlusDelt, V2tPlusDelt, W2tPlusDelt, U3tPlusDelt,
                 V3tPlusDelt, W3tPlusDelt, U4tPlusDelt, V4tPlusDelt, W4tPlusDelt])

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
                    UtVtWttransposetPlusDelt = UtVtWttPlusDelt.transpose()
                    srMatrixUtVtWttransposetPlusDelt = srMatrix.dot(UtVtWttransposetPlusDelt)

                    dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt = Jmatrix.dot(srMatrixUtVtWttransposetPlusDelt)

                    bmatPlusDelt = matrix(
                        [[dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[0, 0] + 1, dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[1, 0],
                          dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[2, 0], 0, 0, 0],
                         [0, 0, 0, dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[3, 0], dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[4, 0] + 1,
                          dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[5, 0]],
                         [dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[3, 0], dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[4, 0] + 1,
                          dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[5, 0], dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[0, 0] + 1,
                          dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[1, 0], dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[2, 0]]])
                    JmatrixsrMatrix = Jmatrix.dot(srMatrix)
                    BMatLintPlusDelt = bmatPlusDelt.dot(JmatrixsrMatrix)
                    CBMatLintPlusDelt = C.dot(BMatLintPlusDelt)
                    BMatLintransposetPlusDelt = BMatLintPlusDelt.transpose()

                    KeLintPlusDelt = KeLintPlusDelt + (
                            (BMatLintransposetPlusDelt.dot(CBMatLintPlusDelt)) * detJ * thick)

                    epsXXtPlusDelt = dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[0, 0] + 1 / 2 * pow(
                        dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[0, 0],
                        2) + 1 / 2 * pow(
                        dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[1, 0], 2) + 1 / 2 * pow(
                        dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[2, 0], 2)
                    epsYYtPlusDelt = dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[4, 0] + 1 / 2 * pow(
                        dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[3, 0],
                        2) + 1 / 2 * pow(
                        dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[4, 0], 2) + 1 / 2 * pow(
                        dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[5, 0], 2)
                    epsXYtPlusDelt = 1 / 2 * (
                            dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[1, 0] + dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[3, 0] +
                            dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[0, 0] * dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[3, 0] +
                            dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[1, 0] * dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[4, 0] +
                            dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[2, 0] * dUdXdVdXdWdXdUdYdVdYdWdYtPlusDelt[5, 0])
                    epsmatrxtPlusDelt = matrix([[epsXXtPlusDelt], [epsYYtPlusDelt], [2 * epsXYtPlusDelt]])

                    StXXStYYStXYtPlusDelt = C.dot(epsmatrxtPlusDelt)

                    BMatNonLintPlusDelt = Jmatrix.dot(srMatrix)

                    CnonLintPlusDelt = matrix([[StXXStYYStXYtPlusDelt[0, 0], 0, 0, StXXStYYStXYtPlusDelt[1, 0], 0, 0],
                                               [0, StXXStYYStXYtPlusDelt[0, 0], 0, 0, StXXStYYStXYtPlusDelt[1, 0], 0],
                                               [0, 0, StXXStYYStXYtPlusDelt[0, 0], 0, 0, StXXStYYStXYtPlusDelt[1, 0]],
                                               [StXXStYYStXYtPlusDelt[1, 0], 0, 0, StXXStYYStXYtPlusDelt[2, 0], 0, 0],
                                               [0, StXXStYYStXYtPlusDelt[1, 0], 0, 0, StXXStYYStXYtPlusDelt[2, 0], 0],
                                               [0, 0, StXXStYYStXYtPlusDelt[1, 0], 0, 0, StXXStYYStXYtPlusDelt[2, 0]]])

                    BMatNonLintransposetPlusDelt = BMatNonLintPlusDelt.transpose()
                    CnonLinBMatNonLintPlusDelt = CnonLintPlusDelt.dot(BMatNonLintPlusDelt)

                    KeNonLintPlusDelt = KeNonLintPlusDelt + (
                            (BMatNonLintransposetPlusDelt.dot(CnonLinBMatNonLintPlusDelt)) * detJ * thick)
                    ritPlusDelt = ritPlusDelt + BMatLintransposetPlusDelt.dot(StXXStYYStXYtPlusDelt) * detJ * thick

            KetPlusDelt = KeLintPlusDelt + KeNonLintPlusDelt

            for k in range(0, DFPN * NPE, 1):
                rigtPlusDelt[ng[0, k], 0] = rigtPlusDelt[ng[0, k], 0] + ritPlusDelt[k, 0]
                for l in range(0, DFPN * NPE, 1):
                    ktgtPlusDelt[ng[0, k], ng[0, l]] = ktgtPlusDelt[ng[0, k], ng[0, l]] + KetPlusDelt[k, l]

        rigredutPlusDelt = zeros((noDof, 1))
        regredutPlusDelt = zeros((noDof, 1))
        reduktgtPlusDelt = zeros((noDof, noDof))
        idofi = 0
        for i in range(0, DFPN * noNode, 1):
            if dof[i] == 1:
                rigredutPlusDelt[idofi, 0] = rigtPlusDelt[i, 0]
                regredutPlusDelt[idofi, 0] = reg[i, 0]
                idofj = 0
                for j in range(0, DFPN * noNode, 1):
                    if dof[j] == 1:
                        reduktgtPlusDelt[idofi, idofj] = ktgtPlusDelt[i, j]
                        idofj = idofj + 1

                idofi = idofi + 1
        sumSqureErrortPlusDelt = 0
        exterLoadSquretPlusDelt = 0
        for i in range(0, noDof, 1):
            mmm = rigredutPlusDelt[i, 0] - regredutPlusDelt[i, 0]
            sumSqureErrortPlusDelt = sumSqureErrortPlusDelt + pow(mmm, 2)
            mmms = regredutPlusDelt[i, 0]
            exterLoadSquretPlusDelt = exterLoadSquretPlusDelt + pow(mmms, 2)

        conv = sumSqureErrortPlusDelt / (1 + exterLoadSquretPlusDelt)
        #print(conv)


        if conv > .004:
            iterationParameter = 0
            sdd = (regredutPlusDelt - rigredutPlusDelt)
            ffd = linalg.inv(reduktgtPlusDelt)
            delu = ffd.dot(sdd)
            glodelu = zeros((DFPN * noNode, 1))
            idofi = 0
            for i in range(0, DFPN * noNode, 1):
                if dof[i] == 1:
                    glodelu[i, 0] = delu[idofi, 0]  ####
                    idofi = idofi + 1
            utPlusDeltglobal = utPlusDeltglobal + glodelu
        else:
            iterationParameter = 1

    print(rigredutPlusDelt)
    print(conv)

    print(regredutPlusDelt)
    print(111)
    print(utPlusDeltglobal)
    print(222)

    outsheet.write(0, 3 * t, "rig")
    outsheet.write(0, 3 * t + 1, "reg")
    outsheet.write(0, 3 * t + 2, "utPlusDeltglobal")
    idofi = 0
    for i in range(0, DFPN * noNode):
        # outsheet.write(i + 1, 0, p1[0, i])
        # outsheet.write(i + 1, 1, p2[0, i])
        outsheet.write(i + 1, 3 * t + 2, utPlusDeltglobal[i, 0])
        if dof[i] == 1:
            outsheet.write(i + 1, 3 * t, rigredutPlusDelt[idofi, 0])
            outsheet.write(i + 1, 3 * t + 1, regredutPlusDelt[idofi, 0])
            idofi = idofi + 1

    utxxglobal = zeros(noNode)
    utyyglobal = zeros(noNode)
    utzzglobal = zeros(noNode)

    for i in range(0, noNode, 1):
        outsheet2.write(i, 3 * t, utPlusDeltglobal[3 * i, 0])
        outsheet2.write(i, 3 * t + 1, utPlusDeltglobal[(3 * i) + 1, 0])
        outsheet2.write(i, 3 * t + 2, utPlusDeltglobal[(3 * i) + 2, 0])

outworkbook.close()



