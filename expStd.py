import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

def generateRandomElements(n, av_low, av_high, dur_low, dur_high):
    popInd = np.arange(n)
    popAV = np.random.randint(low = av_low, high = av_high, size = n)
    popDur = np.random.randint(low = dur_low, high = dur_high, size = n)
    surrChrYrs = np.random.randint(low = 1, high = 5, size = n) #hard coded so i get 2,4,6, and 8
    decRandDraw = np.random.random(size = n)
    df = pd.DataFrame(data = {'accountValue': popAV,
                                    'duration':popDur,
                                    'chrYrs':surrChrYrs,
                                    'randDraw':decRandDraw}, index = popInd)
    return df

def deriveAttributes(df):
    df['chrYrs'] = df.chrYrs.apply(lambda x: x*2)
    df['yrsLeft'] = df.chrYrs - df.duration
    df['yrsLeft'] = df.yrsLeft.apply(lambda x: np.max(x, 0))
    df['charge'] = df.yrsLeft / (df.chrYrs * 11)
    df['amtCharge'] = df.charge * df.accountValue
    return df


def simulateDec(df, distB):
    df['decrement'] = df.apply(lambda x :testDecrement(x['probDec'], x['randDraw'], distB), axis = 1)
    return df

def testDecrement(prob, draw, distortion):
    prob = prob + np.random.rand()*distortion
    if draw < prob:
        return 1
    else:
        return 0
    return 0

def calculateProb(x, y):
    b0 = -2.15
    b1 = -0.2
    b2 = -0.00001
    g = math.exp(b0 + b1*x + b2*(x*y))
    return g/(1+g)


def buildSynthPop(n, av_low, av_high, dur_low, dur_high, simBias):
    synthPop = generateRandomElements(n, av_low, av_high, dur_low, dur_high)
    synthPop = deriveAttributes(synthPop)
    synthPop['probDec'] = synthPop.apply(lambda x: calculateProb(x['yrsLeft'], x['amtCharge']), axis = 1)
    synthPop = simulateDec(synthPop, simBias)
    return synthPop

def process_synthExp(sp_df):
    simExp = sp_df.groupby(['chrYrs', 'duration']).agg({'charge':pd.Series.count, 'decrement':np.sum, 'probDec':np.mean})
    simExp = sp_df.groupby(['chrYrs', 'duration']).agg({'charge':pd.Series.count, 'decrement':np.sum, 'probDec':np.mean})
    simExp.rename(index=str, columns={"charge":"totalPop","decrement":"lapses", "probDec":"expected_lapseRate"}, inplace = True)
    simExp['actual_lapseRate'] = simExp.apply(lambda q: q['lapses']/q['totalPop'], axis = 1)
    simExp['subject'] = 1
    return simExp
