import pandas as pd
from enum import Enum
import numpy as np
import scipy
from numpy import sqrt, sin, cos, tan, pi
from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

'''Water'''
density = 997
dynamicViscosity = 0.0008891
kinematicViscosity = 8.917*10**-7
TotalMassFlowRate = 1.3

'''Can be Set'''
voluteThickness = 0.005
discThickness = 0.0007
discSpacing = 0.0002
wallSpace = 0.004


def derivedAngle(vSpace, vIRadius, vRadius):
    degree = np.arctan(2*vSpace*vRadius/(vIRadius**2))
    return 0.5*pi - degree


class flowParameters():
    def __init__(self, innerRadius, outerRadius, discSpacing, discThickness, numberSpacing,
                 voluteThickness, voluteWallSpace, upperClearance,
                 totMassFlowRate, density, RPM, k_Width_h0=0.798):
        self.innerRadius = innerRadius
        self.outerRadius = outerRadius
        self.discSpacing = discSpacing
        self.discThickness = discThickness
        self.voluteWallSpace = voluteWallSpace
        self.upperClearance = upperClearance
        self.numberSpacing = numberSpacing
        self.massFlowRate = totMassFlowRate
        self.density = density

        self.voluteSpace = numberSpacing*discSpacing + \
            (numberSpacing-1)*discThickness
        self.totalVoluteSpace = 2*discThickness + \
            self.voluteSpace + 2*self.voluteWallSpace
        self.h0 = k_Width_h0*self.totalVoluteSpace

        self.r0 = self.outerRadius + self.upperClearance + voluteThickness + self.totalVoluteSpace/2 + \
            cos(np.arcsin((self.totalVoluteSpace - self.h0)/self.h0))*self.h0

        # formula
        self.inletAngle = derivedAngle(self.voluteSpace, self.h0, self.r0)
        self.vRadial, self.vTheta = flowParameters.velocityInlet(self)
        # end

        self.omega = RPM*2*pi/60

        self.DH = 2*self.discSpacing
        self.massFlowRatePD = self.massFlowRate/self.numberSpacing
        self.volumeFlowRatePD = self.massFlowRatePD/density

        self.tipVelocity = self.omega*self.outerRadius
        self.relativeTipTangential = (
            self.vTheta - self.tipVelocity)/self.tipVelocity
        self.innerOuterRatio = self.innerRadius/self.outerRadius

        self.reynoldM = self.massFlowRatePD / \
            (pi*self.outerRadius*dynamicViscosity)
        self.reynoldMS = self.reynoldM * self.DH / self.outerRadius

    def velocityInlet(self):
        effectiveArea = 2*pi * \
            (self.outerRadius+self.upperClearance)*self.voluteSpace
        vRadial = self.massFlowRate/(effectiveArea*self.density)
        vTheta = vRadial/tan(self.inletAngle)
        vRadialDisc = vRadial*self.outerRadius / \
            (self.outerRadius-self.upperClearance)
        return -vRadialDisc, vTheta


def bothODE(y, x, instance):
    y0, y1 = y
    firstTerm = -2
    secondTerm = ((48/instance.reynoldMS)*x - 1/x)*y0
    firstSolution = firstTerm + secondTerm
    secondSolution = (
        2/x)*((instance.vRadial/(x*instance.tipVelocity))**2 + (y0+x)**2)
    return [firstSolution, secondSolution]


def rotorEff(firstAnswer, rs, instance):
    return (1 - (firstAnswer[-1] + instance.innerOuterRatio)*instance.innerOuterRatio
            / (firstAnswer[0] + 1))


def power(firstAnswer, rs, instance):
    firstAnswerFlip = np.squeeze(np.flip(firstAnswer))
    rsFlip = (np.flip(rs)*instance.outerRadius)

    constantTerm = (2*pi/instance.discSpacing) * \
        (6*dynamicViscosity*instance.tipVelocity)
    integrateTerm = firstAnswerFlip*np.power(rsFlip, 2)
    return 2*constantTerm*scipy.integrate.simps(integrateTerm, x=rsFlip)*instance.omega*instance.numberSpacing


'''base model'''
rotorOuter = 0.073
rotorInner = 0.3*rotorOuter

KJ = flowParameters(rotorInner, rotorOuter, discSpacing, discThickness, 3,
                    0.005, wallSpace, 0,
                    TotalMassFlowRate, density, 750)
firstODEinitial, secondODEinitial = KJ.relativeTipTangential, 0
rs = np.linspace(1, KJ.innerOuterRatio, 100)
sol = odeint(bothODE, [firstODEinitial, secondODEinitial], rs, args=(KJ,))

'''
print(f"Disc Number:\t{KJ.numberSpacing+1}")
print(f"w full:\t\t{KJ.totalVoluteSpace} m")

print(f"w is:\t\t{KJ.voluteSpace} m")
print(f"r0 is:\t\t{KJ.r0} m")
print(f"Outer radius:\t{KJ.outerRadius} m")
print(f"h is:\t\t{KJ.h0} m")
print(f"Angle:\t\t{KJ.inletAngle*180/pi} dg from tangent line")

print(f"R ratio:\t{KJ.innerOuterRatio}")
print(f"Vr:\t\t{KJ.vRadial} \t\tVt:{KJ.vTheta}")

print(f"Reynold:\t{KJ.reynoldM}")
print(f"Reynold*:\t{KJ.reynoldMS}")
print(f"W0:\t\t{KJ.relativeTipTangential}")

print(f"Power:\t\t{power(sol[:,0], rs, KJ)} W")
'''

maxRotorOuter = 0.164

nDiscRange = np.arange(1, 17, 1)
discSpacingRange = np.arange(0.0002, 0.001, 0.0001)
kFactorRange = np.linspace(1, 5, 50)
rpmRange = np.linspace(20, 2500, 100)


'''for i in range(len(discSpacingRange)):'''
for i in range(2):
    bSpacing = discSpacingRange[i]
    hiOutput = 0

    powerStorageStore = []
    for j in range(len(nDiscRange)):
        powerStorage = np.zeros([len(rpmRange), len(kFactorRange)])
        X, Y = np.meshgrid(rpmRange, kFactorRange)

        discNumberSpacing = nDiscRange[j]

        for k in range(len(rpmRange)):
            effectiveRPM = rpmRange[k]
            for l in range(len(kFactorRange)):
                maxRotorOuterCase = maxRotorOuter/kFactorRange[l]

                '''innerRadius, outerRadius, discSpacing, discThickness, numberSpacing,
                voluteThickness, voluteWallSpace, upperClearance,
                totMassFlowRate, density, RPM'''

                KJ = flowParameters(0.3*maxRotorOuterCase, maxRotorOuterCase, bSpacing, discThickness, discNumberSpacing,
                                    0.005, wallSpace, 0,  # 0 for maximum possible power output and efficiency
                                    TotalMassFlowRate, density, effectiveRPM)
                firstODEinitial, secondODEinitial = KJ.relativeTipTangential, 0
                rs = np.linspace(1, KJ.innerOuterRatio, 100)
                sol = odeint(bothODE, [firstODEinitial,
                                       secondODEinitial], rs, args=(KJ,))
                powerStorage[k, l] = power(sol[:, 0], rs, KJ)
                if powerStorage[k, l] > hiOutput:
                    hiOutput = powerStorage[k, l]
        powerStorageStore.append(powerStorage)

    '''plt.rcParams["figure.figsize"] = 30, 24'''

    X, Y = np.meshgrid(rpmRange, kFactorRange)
    powerStorageStore = np.array(powerStorageStore)
    fig, axs = plt.subplots(4, 4, figsize=(10, 12))
    levels = np.arange(0, 100, 10)
    countX, countY = 0, 0
    for q in range(len(powerStorageStore)):
        if(countY == 4):
            countX += 1
            countY = 0
        cs = axs[countX, countY].contourf(
            X, Y, powerStorageStore[q].transpose(), levels=levels, extend='both')
        axs[countX, countY].set_title(f'n = {q+2}')
        axs[countX, countY].tick_params(axis='x')
        axs[countX, countY].tick_params(axis='y')
        countY += 1

    plt.tight_layout()
    cbar = plt.colorbar(cs, ax=axs)
    cbar.ax.set_ylabel('Power Output (W)')
    '''ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=25)'''
    print(f'Done {i}')
plt.show()
