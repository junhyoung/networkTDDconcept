import numpy as np
import matplotlib.pyplot as plt

FreqReUse = 9
NoUpLink = 12
NtwSizeA = 2000
NtwSizeB = -2000
PlusShift = 12000
MinusShift = -12000
No_Iterations = 1000
SIR = np.zeros((1, NoUpLink))

for Loop in range(0, No_Iterations):
    SubY = np.random.uniform(NtwSizeB, NtwSizeA, size=[FreqReUse, NoUpLink])
    SubX = np.random.uniform(NtwSizeB, NtwSizeA, size=[FreqReUse, NoUpLink])

    Cell_x0 = SubX[0, :]
    Cell_y0 = SubY[0, :]

    Cell_x1 = SubX[1, :]
    Cell_y1 = SubY[1, :] + PlusShift

    Cell_x2 = SubX[2, :] + PlusShift
    Cell_y2 = SubY[2, :] + PlusShift

    Cell_x3 = SubX[3, :] + PlusShift
    Cell_y3 = SubY[3, :]

    Cell_x4 = SubX[4, :] + PlusShift
    Cell_y4 = SubY[4, :] + MinusShift

    Cell_x5 = SubX[5, :]
    Cell_y5 = SubY[5, :] + MinusShift

    Cell_x6 = SubX[6, :] + MinusShift
    Cell_y6 = SubY[6, :] + MinusShift

    Cell_x7 = SubX[7, :] + MinusShift
    Cell_y7 = SubY[7, :]

    Cell_x8 = SubX[8, :] + MinusShift
    Cell_y8 = SubY[8, :] + PlusShift

    ShiftX = np.array([Cell_x0, Cell_x1, Cell_x2, Cell_x3,
                   Cell_x4, Cell_x5, Cell_x6, Cell_x7, Cell_x8])

    ShiftY = np.array([Cell_y0, Cell_y1, Cell_y2, Cell_y3,
                   Cell_y4, Cell_y5, Cell_y6, Cell_y7, Cell_y8])

    Dist = np.sqrt(ShiftX ** 2 + ShiftY ** 2)

    NormalDistribution = np.random.randn(FreqReUse, NoUpLink)
    mu = 0
    SD = 6
    LogNormal = mu + SD * NormalDistribution

    nd = np.random.randn(10)  # 10개의 숫자에 대해서 정규분포
    ud = np.random.uniform(-1, 1, 10)  # -1에서 1까지의 10개의 숫자

    plus = np.where(nd > 1)
    minus = np.where(nd < -1)

    plt.figure(1)
    plt.plot(ShiftX, ShiftY, 'bo')

    mu = 0;
    SD = 6;

    LogNormal = mu + SD * NormalDistribution

    LogNormalP = 10 ** (LogNormal / 10) / (Dist ** 4)

    Ps = 10 ** (LogNormal[0, :] / 10) / Dist[0, :] ** 4  # 시그널
    PI1 = 10 ** (LogNormal[1, :] / 10) / Dist[1, :] ** 4  # 인터피어러스 파워 PI
    PI2 = 10 ** (LogNormal[2, :] / 10) / Dist[2, :] ** 4
    PI3 = 10 ** (LogNormal[3, :] / 10) / Dist[3, :] ** 4
    PI4 = 10 ** (LogNormal[4, :] / 10) / Dist[4, :] ** 4
    PI5 = 10 ** (LogNormal[5, :] / 10) / Dist[5, :] ** 4
    PI6 = 10 ** (LogNormal[6, :] / 10) / Dist[6, :] ** 4
    PI7 = 10 ** (LogNormal[7, :] / 10) / Dist[7, :] ** 4
    PI8 = 10 ** (LogNormal[8, :] / 10) / Dist[8, :] ** 4

    PI = PI1 + PI2 + PI3 + PI4 + PI5 + PI6 + PI7 + PI8
    matrixP = np.array([Ps, PI1, PI2, PI3, PI4, PI5, PI6, PI7, PI8])
    SIRn = Ps/PI

    SIRdB=10*np.log10(SIRn)

    SIR=np.vstack((SIR,SIRdB))

PI0SUM = sum(matrixP[1:9, 0])
PI1SUM = sum(matrixP[1:9, 1])
PI2SUM = sum(matrixP[1:9, 2])
PI3SUM = sum(matrixP[1:9, 3])
PI4SUM = sum(matrixP[1:9, 4])
PI5SUM = sum(matrixP[1:9, 5])
PI6SUM = sum(matrixP[1:9, 6])
PI7SUM = sum(matrixP[1:9, 7])
PI8SUM = sum(matrixP[1:9, 8])
PI9SUM = sum(matrixP[1:9, 9])
PI10SUM = sum(matrixP[1:9, 10])
PI11SUM = sum(matrixP[1:9, 11])

P = 10 ** (LogNormal / 10)
P[0, 0]
Dist[0, 0]

# SIR = Signal to interpeareance Ratio
