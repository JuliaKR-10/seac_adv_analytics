import pandas as pd
import numpy as np
from datetime import datetime as dt
from string import Template
from pandas.tseries.offsets import *

# generates start date as just a seed date plus the policy number in days rolled to the following business day
def genStartDt(polNo, seedDate):
    offset = Bday() #to have all entry at business day -- smoother
    return offset.rollforward(seedDate + pd.Timedelta(Day(polNo)))

#set industry, allow for simulating that only in edu at first
def genInd(startDate, brkDt):
    education = True
    if(startDate > brkDt):
        if(np.random.uniform()<0.4):
            education = False
    return education

#generate tenure in months
def genTenure(industry): #currently true false education or other; returns tenure in months
    if(industry):
        return np.random.negative_binomial(n=46, p=.25) #mean of 104 months
    else:
        return np.random.binomial(n=206, p=0.7) #decide mean 12yrs(144 mos) and p = 0.7
    return 0

#generates tenure and calcuates policy end date
def calcEndDt(sd, ind):
    offset = BMonthEnd()
    ed = offset.rollforward(sd + pd.Timedelta(Day(genTenure(ind)*30)))
    return min(ed, pd.Timestamp(dt.now()))

# generates the person's DOB
def genDOB(sd):
    minInitAge = 22*365
    maxInitAge = 55*365
    rd = pd.Timedelta(np.random.randint(low = minInitAge, high = maxInitAge), unit = 'd')
    return sd - rd

def genGen():
    if np.random.uniform() < 0.5:
        return True
    else:
        return False

def getAge(d1,d2):
    return (d2-d1)/pd.Timedelta(1,'Y')

def genIncome(sd,dob, edu):
    age = getAge(dob, sd)
    inc = 50000
    if edu:
        inc = inc + (age-22)*1500 +np.random.uniform(low = 0, high = 500)
    else:
        inc = inc + (age-22)* 1000 +np.random.uniform(low = 0, high = 1200)
    return inc

# generates sythetic population of size n with seed date sd and nonedu start date of bkDt
def buildRetPop(n, sd, bkDt):
    polNos = np.arange(n)
    stDts = pd.bdate_range(sd, periods=n)
    df = pd.DataFrame(data={'startDate': stDts}, index=polNos)
    df['eduMkt'] = df.apply(lambda x : genInd(x['startDate'], bkDt), axis=1)
    df['endDt'] = df.apply(lambda x : calcEndDt(x['startDate'], x['eduMkt']), axis=1)
    df['dob'] = df.apply(lambda x: genDOB(x['startDate']), axis = 1)
    df['isMale'] = df.apply(lambda x: genGen(), axis = 1)
    df['inc'] = df.apply(lambda x: genIncome(x['startDate'], x['dob'], x['eduMkt']), axis = 1)
    return df


#function to generate the non-auto contributions
def getOther(sd, ed, prob, premType):
    posDts = pd.date_range(start = sd, end = ed, freq = 'B')
    rd = np.random.uniform(size = posDts.size)
    rdf = pd.DataFrame(data = {'randomDraw':rd}, index = posDts)
    ocdf = rdf.drop(rdf[rdf.randomDraw>prob].index)
    ocdf['premType'] = premType
    return ocdf.drop(['randomDraw'], axis = 1).reset_index()

#function to generate the auto pay with probabalistic summer pay term
def getAuto(sd, ed, probSP):
    posDts = pd.date_range(start = sd, end = ed, freq = 'SM')
    month = posDts.month
    rd = np.random.uniform(size = posDts.size)
    rdf = pd.DataFrame(data = {'randomDraw':rd, 'month':month}, index = posDts)
    ocdf = rdf.drop(rdf[rdf.randomDraw>0.97].index)
    ocdf['premType'] = 'auto'
    if(np.random.uniform() < probSP):
        ocdf = ocdf.drop(ocdf[ocdf.month == 7].index)
        ocdf = ocdf.drop(ocdf[ocdf.month == 8].index)
    return ocdf.drop(['randomDraw','month'],axis=1).reset_index()


# probability that a teach will have 10 month payments instaed of 12
def getSummerPayProb(ind):
    if(ind):
        return np.random.beta(a = 2, b = 5) #mean ~ 0.3 w/ fat lh tail and long rh tail bounded b/t 0 and 1
    else:
        return 0
    return 0

# simulate termination transaction date as autopay end date plus a random number of days
# the number of days is a draw from a poisson dist where the mean is a function of age, industry,
# rollover behavior, ad hoc behavior, and income
def getTermTrx(sd, ed, dob, edu, income, rollovrDummy, lastAdHoc):
    age = getAge(dob, ed)
    yrsAdHoc = getAge(lastAdHoc, ed)
    eduD = 0
    if edu:
        eduD = 1
    # simulate the probability that there is a term date as a function of age
    lmda = ((2.5 * 360)
        + -30.0 *(age - 55)
        + -2.5 * income/1000
        + 180 * eduD
        + -90 * yrsAdHoc
        + 180 * rollovrDummy
        + 360 * np.random.normal())
    termDt = ed + pd.Timedelta(max(np.random.poisson(lam = max(lmda, 7.0)),1.0), 'd')
    return termDt

# probability that the ph will do a rollover on any given day
def getRollovrProb(t):
    return np.random.uniform(low = 0, high = .002)/t


# generate indivudual probability for ad hoc contribution on any given day
def getAdHocProb(eduInd, surgeInd):
    prb = 0
    alphaAdj = 0

    #non edu have a higher probability of drop in ad hocs
    if eduInd:
        alphaAdj = -0.5

    if surgeInd:
        alpha = 3 + alphaAdj
        prb = np.random.beta(a=alpha, b = 1) # increased likelihood that there are drop ins during surge
    else:
        alpha = 1 + alphaAdj
        prb = np.random.beta(a=alpha, b = 20) #much less likely to have drop in during the period
    return (1-(1-prb)**(1/60))

# generate the transaction table for a given policy
def genPolTrx(polNo, sd, ed, edu, tenure, dob, income):
    offset_prop = BYearBegin()
    sd = offset_prop.rollforward(sd)
    ahSurg_sd = pd.Timestamp('2014-10-01')
    ahSurg_ed = pd.Timestamp('2015-01-01')
    dfa = getAuto(sd,ed,getSummerPayProb(edu))
    df_adhoc = getOther(sd,ed, getAdHocProb(edu, False), 'adhoc')
    lstAdHoc = df_adhoc['index'].max()
    if(ed >ahSurg_ed):
        df_adhoc = df_adhoc.append(getOther(ahSurg_sd,ahSurg_ed, getAdHocProb(edu, True), 'adhoc'))
    df_rollovr = getOther(sd,ed, getRollovrProb(tenure), 'rollover')

    if df_rollovr.shape[0] > 0:
        rlovrD = 1
    else:
        rlovrD = 0

    odf = dfa.append(df_adhoc).append(df_rollovr)
    odf['polNo'] = polNo
    odf['educInd'] = edu
    odf = odf.rename(index = str, columns={"index":"trxDt"})

    if ed < (pd.Timestamp(dt.now()) - pd.Timedelta('5d')):
        termTrxDt = getTermTrx(sd,ed,dob,edu,income,rlovrD,lstAdHoc)
        if termTrxDt < (pd.Timestamp(dt.now()) - pd.Timedelta('4d')):
            odf = odf.append({'educInd':edu,'polNo':polNo, 'premType':'termination','trxDt':termTrxDt}, ignore_index = True)
    return odf


def buildTrx(df):
    cols = ['polNo', 'educInd','trxDt', 'premType']
    trx_df = pd.DataFrame(columns = cols)
    for index, row in df.iterrows():
        trx_df = trx_df.append(genPolTrx(index,
                                row['startDate'], row['endDt'], row['eduMkt'],
                                int((row['endDt']-row['startDate'])/pd.Timedelta(1,'Y')),row['dob'],row['inc']))
    trx_df['units'] = 1
    return trx_df
