import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from talib import abstract




class SAR_agent:
    def __init__(self, paradict: dict):
        self.paradict = paradict

    def generate_signal(self, time_bar):
        sar = abstract.SAR(time_bar, acceleration=self.paradict['sar_acce'], maximum=self.paradict['sar_max'])
        macd = abstract.MACD(time_bar, fastperiod=self.paradict['macd_fast'], slowperiod=self.paradict['macd_slow'], signalperiod=self.paradict['macd_period'])
        up = (sar > time_bar.close).astype(int)
        up_diff = -up.diff()
        
        #down = (sar < time_bar.close).astype(int)
        #down_diff = -down.diff()
        
        macd_roll_buy = (macd.macdhist < 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        macd_roll_sell  = (macd.macdhist > 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        # this returns a signal array for every timestamp
        macd_diff_buy = ((macd.macdhist).diff() > 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        macd_diff_sell = ((macd.macdhist).diff() < 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        
        #nega_macd = (macd.macd < -5).astype(int).rolling(self.paradict['roll_length'], min_periods=1).sum()

        sar_sell = up.diff().rolling(self.paradict['roll_length'], min_periods=1).sum()
        buy = (up_diff == 1) & (macd_roll_buy == 2)  & (macd_diff_buy == 2) #& (nega_macd > 0)
        sell = (sar_sell > 0) & (macd_roll_sell == 2)  & (macd_diff_sell == 2)
        return buy, sell

    def print_stat(self, real_buy, real_sell, values, positions, time_bar):
        fig, ax = plt.subplots(5, figsize=(20, 10))
        macd = abstract.MACD(time_bar, fastperiod=self.paradict['macd_fast'], slowperiod=self.paradict['macd_slow'], signalperiod=self.paradict['macd_period'])
        vol_avg = time_bar.volume.rolling(self.paradict['roll_length'], min_periods=1).mean()
        sar = abstract.SAR(time_bar, accelation=self.paradict['sar_acce'], maximum=self.paradict['sar_max'])
        ax[0].plot(time_bar.close)
        ax[0].plot(time_bar.close[real_buy], 'v')
        ax[0].plot(time_bar.close[real_sell], '^')
        ax[0].plot(sar, '.', mfc='none')

        ax[1].plot(macd.macd, color='orange')
        ax[1].plot(macd.macdsignal, 'b')
        ax[1].hlines(y=0, xmin=time_bar.index[0], xmax=time_bar.index[-1], linestyles='--', color='brown')
        ax[1].plot(macd.macdhist, color='c', linestyle=':')
        ax[1].plot(macd.macdhist[real_buy], 'v', color='orange')
        ax[1].plot(macd.macdhist[real_sell], '^', color='g')
        ax[1].legend(['DIF', 'DEA', 'MACD'])

        ax[2].plot(time_bar.volume, color='r', linestyle=':')
        ax[2].plot(vol_avg, 'b')
        ax[2].plot(time_bar.volume[real_buy], 'v', color='orange')
        ax[2].plot(time_bar.volume[real_sell], '^', color='g')
        ax[2].legend(['volume', 'vol_5min'])

        ax[3].plot(values)
        ax[3].legend(['pnl'])

        ax[4].plot(positions)
        ax[4].legend(['position'])
        return fig


def print_stats_from_result(r_np, date_list):
    print(r_np.shape)

    # stat on each trading day
    res_by_date = []
    for i in range(len(date_list)):
        res_by_date.append(r_np[:, i].sum())


    
    res_by_date = pd.Series(res_by_date, index=date_list)
    print(res_by_date)
    min_day = res_by_date.sort_values(ascending=True).index[0]
    max_day = res_by_date.sort_values(ascending=False).index[0]
    print("max_day:", max_day, 'min_day', min_day)
    plt.hist(res_by_date, 50, density=True, alpha=0.75)
    plt.show()
    print(pd.Series(res_by_date).describe())

    # stat for each stock * day with trades
    res = r_np.flatten()
    print(pd.Series(res).describe())
    print(f"Zero ratio {sum(res == 0) / len(res)}")
    print(f"Positive ratio {sum(res > 0) / len(res)}")

    res_without_0 = [r for r in res if r != 0]
    plt.hist(res_without_0, 50, range=(-2000, 2000), density=True, alpha=0.75)
    plt.show()
    print(pd.Series(res_without_0).describe())


def sar_stra_v3(time_bar, paradict, agent=SAR_agent, print_fig=False):
    agent = agent(paradict)

    buy, sell_sig = agent.generate_signal(time_bar)

    def transform(row):
        p = ((row > 0) & (row <= time_bar.close)).astype(int)
        if sum(p) > 0:
            return row[p == 1].index[0]
        elif sum(row != 0) > 0:
            return row[row != 0].index[-1]
        else:
            return row.index[-1]

    def transform2(row):
        p = ((row > 0) & (row >= time_bar.close)).astype(int)
        if sum(p) > 0:
            return row[p == 1].index[0]
        elif sum(row != 0) > 0:
            return row[row != 0].index[-1]
        else:
            return row.index[-1]
        
    def transform3(row,buy_loc):
        p = ((row == 1) & (row.index >= buy_loc)).astype(int)
        if sum(p) > 0:
            return row[p == 1].index[0]
        elif sum(row != 0) > 0:
            return row[row != 0].index[-1]
        else:
            return row.index[-1]

    def generate_sell_signal(df_input, buy_signal,sell_signal,profit,loss):
        df = df_input.copy()
        df['buy'] = buy_signal
        buy_loc = buy_signal[buy_signal == 1].index[0]
        # init a sell-price vector
        df = df.apply(func=lambda x: x['close'] * paradict['profit_coef'] if x['buy'] else np.nan, axis=1)

        # generate a sell-price table with a drift
        df = pd.DataFrame(np.diag(df), columns=df.index)
        df = df.drop(df.index[df.sum() == 0], axis=0).replace(0, np.nan)

        df = df.apply(func=lambda x: x.ffill(axis=0, inplace=False), axis=1)
        sell_loc = df.apply(func=transform, axis=1)

        df2 = df_input.copy()
        df2['buy'] = buy_signal
        # init a sell-price vector
        df2 = df2.apply(func=lambda x: x['close'] / paradict['stoploss_ratio'] if x['buy'] else np.nan, axis=1)

        # generate a sell-price table with a drift
        df2 = pd.DataFrame(np.diag(df2), columns=df2.index)
        df2 = df2.drop(df2.index[df2.sum() == 0], axis=0).replace(0, np.nan)

        df2 = df2.apply(func=lambda x: x.ffill(axis=0, inplace=False), axis=1)
        sell_loc2 = df2.apply(func=transform2, axis=1)
        
        if sell_signal[((sell_signal== 1) & (sell_signal.index > buy_loc))].empty :
            sell_loc3 = sell_signal.index[-1]
        else: 
            sell_loc3= sell_signal[((sell_signal== 1) & (sell_signal.index > buy_loc))].index[0]
        sell = pd.Series(data=0, index=df_input.index)

        if (sell_loc.values[0] <= sell_loc2.values[0]) and (sell_loc.values[0] <= sell_loc3):
            sell.loc[sell_loc] = 1
            profit += 1
        elif((sell_loc2.values[0] <= sell_loc.values[0]) and (sell_loc2.values[0] <= sell_loc3)):
            sell.loc[sell_loc2] = 1
            loss += 1
        else:
            sell.loc[sell_loc3] = 1
        return sell,profit,loss
    
    
    if paradict['constraint'] == False:
        profit = 0
        loss = 0
        sell = pd.Series(data=0, index=time_bar.index)
        for i in buy[buy == 1].index:
            buy_signal = pd.Series(data=0, index=buy.index)
            buy_signal.loc[i] = 1
            sell_signal,profit,loss = generate_sell_signal(time_bar, buy_signal,sell_sig,profit,loss)
            sell += sell_signal

    # position
        pos_rec = pd.Series(buy.astype(int))
        for i in range(1, len(pos_rec)):
            if (pos_rec[i-1] == 0) and sell[i]:
                pos_rec[i] = pos_rec[i-1]
            else:
                pos_rec[i] = pos_rec[i-1] + (1 if buy[i] and not sell[i] else -int(sell[i]))
                
    
    real_buy = pos_rec.diff() > 0
    real_sell = pos_rec.diff() < 0

    # from trading

    fee = paradict["fee"]
    c = (-(pos_rec.diff())*time_bar.close)
    c[c > 0] = c[c > 0] * (1 - fee)
    c[c < 0] = c[c < 0] * (1 + fee)
    c = c.cumsum()

    b = (pos_rec)*time_bar.close

    value = (b+c)

    if print_fig:
        fig = agent.print_stat(real_buy, real_sell, value, pos_rec, time_bar)
        fig.show()
    return value.iloc[-1], profit, loss, value


def sar_stra_v4(time_bar, paradict, agent=SAR_agent,last_pos=0,cost=0, print_fig=False):
    agent = agent(paradict)

    buy, sell_sig = agent.generate_signal(time_bar)

    def transform(row):
        p = ((row > 0) & (row <= time_bar.close)).astype(int)
        if sum(p) > 0:
            return row[p == 1].index[0]
        elif sum(row != 0) > 0:
            return row[row != 0].index[-1]
        else:
            return row.index[-1]

    def transform2(row):
        p = ((row > 0) & (row >= time_bar.close)).astype(int)
        if sum(p) > 0:
            return row[p == 1].index[0]
        elif sum(row != 0) > 0:
            return row[row != 0].index[-1]
        else:
            return row.index[-1]
        
    def transform3(row,buy_loc):
        p = ((row == 1) & (row.index >= buy_loc)).astype(int)
        if sum(p) > 0:
            return row[p == 1].index[0]
        elif sum(row != 0) > 0:
            return row[row != 0].index[-1]
        else:
            return row.index[-1]

    def generate_sell_signal(df_input, buy_signal,sell_signal,profit,loss):
        df = df_input.copy()
        df['buy'] = buy_signal
        buy_loc = buy_signal[buy_signal == 1].index[0]
        # init a sell-price vector
        df = df.apply(func=lambda x: x['close'] * paradict['profit_coef'] if x['buy'] else np.nan, axis=1)

        # generate a sell-price table with a drift
        df = pd.DataFrame(np.diag(df), columns=df.index)
        df = df.drop(df.index[df.sum() == 0], axis=0).replace(0, np.nan)

        df = df.apply(func=lambda x: x.ffill(axis=0, inplace=False), axis=1)
        sell_loc = df.apply(func=transform, axis=1)

        df2 = df_input.copy()
        df2['buy'] = buy_signal
        # init a sell-price vector
        df2 = df2.apply(func=lambda x: x['close'] / paradict['stoploss_ratio'] if x['buy'] else np.nan, axis=1)

        # generate a sell-price table with a drift
        df2 = pd.DataFrame(np.diag(df2), columns=df2.index)
        df2 = df2.drop(df2.index[df2.sum() == 0], axis=0).replace(0, np.nan)

        df2 = df2.apply(func=lambda x: x.ffill(axis=0, inplace=False), axis=1)
        sell_loc2 = df2.apply(func=transform2, axis=1)
        
        if sell_signal[((sell_signal== 1) & (sell_signal.index > buy_loc))].empty :
            sell_loc3 = sell_signal.index[-1]
        else: 
            sell_loc3= sell_signal[((sell_signal== 1) & (sell_signal.index > buy_loc))].index[0]
        sell = pd.Series(data=0, index=df_input.index)

        if (sell_loc.values[0] <= sell_loc2.values[0]) and (sell_loc.values[0] <= sell_loc3):
            sell.loc[sell_loc] = 1
            profit += 1
        elif((sell_loc2.values[0] <= sell_loc.values[0]) and (sell_loc2.values[0] <= sell_loc3)):
            sell.loc[sell_loc2] = 1
            loss += 1
        else:
            sell.loc[sell_loc3] = 1
        return sell,profit,loss
    
    def generate_sell_signal_constraint(df_input, buy_signal,count):
        df = df_input.copy()
        df['buy'] = buy_signal
        # init a sell-price vector
        df = df.apply(func=lambda x: x['close'] * paradict['profit_coef'] if x['buy'] else np.nan, axis=1)

        # generate a sell-price table with a drift
        df = pd.DataFrame(np.diag(df), columns=df.index)
        df = df.drop(df.index[df.sum() == 0], axis=0).replace(0, np.nan)

        df = df.apply(func=lambda x: x.ffill(axis=0, inplace=False), axis=1)
        sell_loc = df.apply(func=transform, axis=1)
        sell = pd.Series(data=0, index=df_input.index)

        df2 = df_input.copy()
        df2['buy'] = buy_signal
        # init a sell-price vector
        df2 = df2.apply(func=lambda x: x['close'] / paradict['stoploss_ratio'] if x['buy'] else np.nan, axis=1)

        # generate a sell-price table with a drift
        df2 = pd.DataFrame(np.diag(df2), columns=df2.index)
        df2 = df2.drop(df2.index[df2.sum() == 0], axis=0).replace(0, np.nan)

        df2 = df2.apply(func=lambda x: x.ffill(axis=0, inplace=False), axis=1)
        sell_loc2 = df2.apply(func=transform2, axis=1)
        
        if sell_loc.values >= sell_loc2.values:
            sell.loc[sell_loc2] = 1
            count +=1
        else:
            sell.loc[sell_loc] = 1
        return sell,count 
    
    if paradict['constraint'] == False:
        profit = 0
        loss = 0
        sell = pd.Series(data=0, index=time_bar.index)
        for i in buy[buy == 1].index:
            buy_signal = pd.Series(data=0, index=buy.index)
            buy_signal.loc[i] = 1
            sell_signal,profit,loss = generate_sell_signal(time_bar, buy_signal,sell_sig,profit,loss)
            sell += sell_signal
        sell.iloc[-1]=0
    # position
        pos_rec = pd.Series(buy.astype(int))+last_pos
        for i in range(1, len(pos_rec)):
            if (pos_rec[i-1] == 0) and sell[i]:
                pos_rec[i] = pos_rec[i-1]
            else:
                pos_rec[i] = pos_rec[i-1] + (1 if buy[i] and not sell[i] else -int(sell[i]))
                
    else:
        sell = pd.Series(data=0, index=time_bar.index)
        count = 0
        buy_contraint=pd.Series(data=0, index=time_bar.index)
        for i in buy[buy == 1].index:
            if count>=3:
                break
            buy_signal = pd.Series(data=0, index=buy.index)
            buy_signal.loc[i] = 1
            buy_contraint += buy_signal
            sell_bar, count = generate_sell_signal_constraint(time_bar, buy_signal,count)
            sell += sell_bar

        # position
        pos_rec = buy_contraint
        for i in range(1, len(pos_rec)):
            if (pos_rec[i-1] == 0) and sell[i]:
                pos_rec[i] = pos_rec[i-1]
            else:
                pos_rec[i] = pos_rec[i-1] + (1 if buy_contraint[i] and not sell[i] else -int(sell[i]))

    real_buy = pos_rec.diff() > 0
    real_sell = pos_rec.diff() < 0

    # from trading

    fee = paradict["fee"]
    c = (-(pos_rec.diff())*time_bar.close)
    c[c > 0] = c[c > 0] * (1 - fee)
    c[c < 0] = c[c < 0] * (1 + fee)
    c = c.cumsum()+cost

    b = (pos_rec)*time_bar.close

    value = (b+c)

    if print_fig:
        fig = agent.print_stat(real_buy, real_sell, value, pos_rec, time_bar)
        fig.show()
    return value.iloc[-1], profit, loss, value,pos_rec,pos_rec.iloc[-1],c.iloc[-1]


def run_sar_backtest_v3(sar_agent, paradict, start_date: str, duration: int, df=pd.DataFrame([])):
    r = []
    date_list = []
    try:
        time_bars = df
        profit = 0
        loss = 0
        tmp_r = []
        value = pd.Series()
        last_val = 0
        init_index = time_bars.index[0]
        for j in range(duration):
            timebar= time_bars[(init_index+(j)*datetime.timedelta(days=1)<=time_bars.index) & (time_bars.index<init_index+(j+1)*datetime.timedelta(days=1))]
            if not timebar.empty and timebar.shape[0] > 100:
                result, pro, lo, val = sar_stra_v3(timebar, paradict, sar_agent)
                tmp_r.append(result if result else 0)
                val += last_val
                last_val += result
                profit += pro
                loss += lo
                value = pd.concat([value,val.dropna()])
            else:
                tmp_r.append(0)
            if len(date_list) < duration:
                date_list.append((time_bars.index[0] + j * datetime.timedelta(days=1)).strftime('%Y%m%d')[2:])
        print(f"It earns {sum(tmp_r)}")
        r.append(tmp_r)
    except e:
        print(f"fail to process, error {str(e)}")

    r_np = np.array([np.array(rr) for rr in r ])
    print_stats_from_result(r_np, date_list)
    print(f"每次胜率：{profit/(profit+loss)}")
    return value,profit/(profit+loss),(profit+loss)


def run_sar_backtest_v4(sar_agent, paradict, start_date: str, duration: int, df=pd.DataFrame([])):
    r = []
    date_list = []
    try:
        time_bars = df
        profit = 0
        loss = 0
        tmp_r = []
        value = pd.Series()
        pos_rec = pd.Series()
        last_val = 0
        last_pos = 0
        init_index = time_bars.index[0]
        c = 0
        for j in range(duration):
            timebar= time_bars[(init_index+(j)*datetime.timedelta(days=1)<=time_bars.index) & (time_bars.index<init_index+(j+1)*datetime.timedelta(days=1))]
            if not timebar.empty and timebar.shape[0] > 100:
                today_val, pro, lo, val, pos,today_pos,c= sar_stra_v4(timebar, paradict, sar_agent,last_pos,c)
                tmp_r.append((today_val-last_val) if (today_val-last_val) else 0)
                last_val = today_val
                last_pos = today_pos
                profit += pro
                loss += lo
                value = pd.concat([value,val.dropna()])
                pos_rec = pd.concat([pos_rec,pos.dropna()])
            else:
                tmp_r.append(0)
            if len(date_list) < duration:
                date_list.append((time_bars.index[0] + j * datetime.timedelta(days=1)).strftime('%Y%m%d')[2:])
        print(f"It earns {sum(tmp_r)}")
        r.append(tmp_r)
    except e:
        print(f"fail to process, error {str(e)}")

    r_np = np.array([np.array(rr) for rr in r ])
    print_stats_from_result(r_np, date_list)
    print(f"每次胜率：{profit/(profit+loss)}")
    return value,profit/(profit+loss),(profit+loss),pos_rec

def find_num_days(start_date: str, end_date: str):
    sd = datetime.datetime.strptime(start_date, '%y%m%d')
    ed = datetime.datetime.strptime(end_date, '%y%m%d')
    return (ed-sd).days


class SAR_Strat:
    def __init__(self, paradict, strat="v3", agent="v0", start_time='200505', duration=60, df=[]):
        self.strat = strat
        self.agent = agent
        self.start_time = start_time
        self.duration = duration
        self.time_bars_all = df
        self.paradict = paradict

    def plotting(self, end_date: str):
        date = find_num_days(start_date=self.start_time, end_date=end_date)
        print(date)
        init_index = self.time_bars_all.index[0]
        time_bars_plot = self.time_bars_all[(init_index+(date)*datetime.timedelta(days=1) <= self.time_bars_all.index) 
                                            & (self.time_bars_all.index < init_index+(date+1) * datetime.timedelta(days=1))]

        agent_dict = {"v0": SAR_agent}

        if self.strat == "v3":
            sar_stra_v3(time_bars_plot, self.paradict, agent_dict.get(self.agent), print_fig=True)
        if self.strat == "v4":
            sar_stra_v4(time_bars_plot, self.paradict, agent_dict.get(self.agent),0,0,print_fig=True)

    def Backtest(self):
        agent_dict = {"v0": SAR_agent}
        if self.strat == "v3":
            return run_sar_backtest_v3(agent_dict[self.agent], self.paradict, self.start_time, self.duration, df=self.time_bars_all)
        if self.strat == "v4":
            return run_sar_backtest_v4(agent_dict[self.agent], self.paradict, self.start_time, self.duration, df=self.time_bars_all)