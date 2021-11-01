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
        # Only Sar: valid the first three rows; Only MACD: valid the middle 4 rows; ALL :all
        up = (sar > time_bar.close).astype(int)
        up_diff = -up.diff()
        sar_sell = up.diff().rolling(self.paradict['roll_length'], min_periods=1).sum()
        
        macd_roll_buy = (macd.macdhist < 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        macd_roll_sell  = (macd.macdhist > 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        macd_diff_buy = ((macd.macdhist).diff() > 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        macd_diff_sell = ((macd.macdhist).diff() < 0).rolling(self.paradict['roll_length'], min_periods=1).sum()
        
        buy = (up_diff == 1) & (macd_roll_buy == 2)  & (macd_diff_buy == 2) 
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

    def generate_sell_signal(df_input, buy_loc,sell_signal):
        df_tmp = df_input.loc[buy_loc:]
        price_open = df_tmp['close'][0]
        sell_sig = sell_signal.loc[buy_loc:]
        idx_close = sell_sig[sell_sig ==1].first_valid_index()
        
        if idx_close == None:
            idx_close = df_tmp.index[-1]
        df_tmp = df_tmp.loc[:idx_close]
        df_sp = df_tmp.loc[df_tmp['close'] >= price_open*paradict['profit_coef']]
        df_sl = df_tmp.loc[df_tmp['close'] <= price_open/paradict['stoploss_ratio']]
    
        if df_sp.empty and df_sl.empty:
            pass
        elif(df_sp.empty):
            idx_close = df_sl.index[0]
        elif(df_sl.empty):
            idx_close = df_sp.index[0]
        else:
            idx_close = min(df_sp.index[0],df_sl.index[0])
        
        sell = pd.Series(data=0, index=sell_signal.index)
        sell.loc[idx_close] = 1
        return sell
    
    sell = pd.Series(data=0, index=time_bar.index)
    for i in buy[buy == 1].index:
        sell += generate_sell_signal(time_bar, i,sell_sig)
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
    return value.iloc[-1], value


def sar_stra_v4(time_bar, paradict, agent=SAR_agent,last_pos=0,cost=0, print_fig=False):
    agent = agent(paradict)

    buy, sell_sig = agent.generate_signal(time_bar)

    def generate_sell_signal(df_input, buy_loc,sell_signal):
        df_tmp = df_input.loc[buy_loc:]
        price_open = df_tmp['close'][0]
        sell_sig = sell_signal.loc[buy_loc:]
        idx_close = sell_sig[sell_sig ==1].first_valid_index()
        
        if idx_close == None:
            idx_close = df_tmp.index[-1]
        df_tmp = df_tmp.loc[:idx_close]
        df_sp = df_tmp.loc[df_tmp['close'] >= price_open*paradict['profit_coef']]
        df_sl = df_tmp.loc[df_tmp['close'] <= price_open/paradict['stoploss_ratio']]
    
        if df_sp.empty and df_sl.empty:
            pass
        elif(df_sp.empty):
            idx_close = df_sl.index[0]
        elif(df_sl.empty):
            idx_close = df_sp.index[0]
        else:
            idx_close = min(df_sp.index[0],df_sl.index[0])
        
        sell = pd.Series(data=0, index=sell_signal.index)
        sell.loc[idx_close] = 1
        return sell
    
    sell = pd.Series(data=0, index=time_bar.index)
    for i in buy[buy == 1].index:
        sell += generate_sell_signal(time_bar, i,sell_sig)
    sell.iloc[-1]=0
    # position
    pos_rec = pd.Series(buy.astype(int))+last_pos
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
    c = c.cumsum()+cost

    b = (pos_rec)*time_bar.close

    value = (b+c)

    if print_fig:
        fig = agent.print_stat(real_buy, real_sell, value, pos_rec, time_bar)
        fig.show()
    return value.iloc[-1], value,pos_rec,pos_rec.iloc[-1],c.iloc[-1]

def sar_stra_v5(time_bar, paradict,last_high,agent=SAR_agent,last_pos=0,cost=0, print_fig=False):
    agent = agent(paradict)

    buy, sell_sig = agent.generate_signal(time_bar)

    def generate_sell_signal(df_input, buy_loc,sell_signal):
        df_tmp = df_input.loc[buy_loc:]
        price_open = df_tmp['close'][0]
        sell_sig = sell_signal.loc[buy_loc:]
        idx_close = sell_sig[sell_sig ==1].first_valid_index()
        
        if idx_close == None:
            idx_close = df_tmp.index[-1]
        df_tmp = df_tmp.loc[:idx_close]
        df_sp = df_tmp.loc[df_tmp['close'] >= price_open*paradict['profit_coef']]
        df_sl = df_tmp.loc[df_tmp['close'] <= price_open/paradict['stoploss_ratio']]
    
        if df_sp.empty and df_sl.empty:
            pass
        elif(df_sp.empty):
            idx_close = df_sl.index[0]
        elif(df_sl.empty):
            idx_close = df_sp.index[0]
        else:
            idx_close = min(df_sp.index[0],df_sl.index[0])
        
        sell = pd.Series(data=0, index=sell_signal.index)
        sell.loc[idx_close] = 1
        return sell
    
    sell = pd.Series(data=0, index=time_bar.index)
    for i in buy[buy == 1].index:
        sell += generate_sell_signal(time_bar, i,sell_sig)
    sell.iloc[-1]=0
    today_high = time_bar.high.max()
    # position
    pos_rec = pd.Series(buy.astype(int))+last_pos
    
    if today_high >= last_high:
        sell_index = time_bar.loc[time_bar['high'] >= last_high].index[0]
        sell.loc[sell_index] += last_pos
        last_pos = 0
    
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
    c = c.cumsum()+cost

    b = (pos_rec)*time_bar.close

    value = (b+c)

    if print_fig:
        fig = agent.print_stat(real_buy, real_sell, value, pos_rec, time_bar)
        fig.show()
    return value.iloc[-1], value,pos_rec,pos_rec.iloc[-1],c.iloc[-1],today_high

def run_sar_backtest_v3(sar_agent, paradict, start_date: str, duration: int, df=pd.DataFrame([])):
    r = []
    date_list = []
    try:
        time_bars = df
        tmp_r = []
        value = pd.Series()
        last_val = 0
        init_index = time_bars.index[0]
        for j in range(duration):
            timebar= time_bars[(init_index+(j)*datetime.timedelta(days=1)<=time_bars.index) & (time_bars.index<init_index+(j+1)*datetime.timedelta(days=1))]
            if not timebar.empty and timebar.shape[0] > 100:
                result, val = sar_stra_v3(timebar, paradict, sar_agent)
                tmp_r.append(result if result else 0)
                val += last_val
                last_val += result
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
    return value


def run_sar_backtest_v4(sar_agent, paradict, start_date: str, duration: int, df=pd.DataFrame([])):
    r = []
    date_list = []
    try:
        time_bars = df
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
                today_val, val, pos,today_pos,c= sar_stra_v4(timebar, paradict, sar_agent,last_pos,c)
                tmp_r.append((today_val-last_val) if (today_val-last_val) else 0)
                last_val = today_val
                last_pos = today_pos
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
    #print(f"每次胜率：{profit/(profit+loss)}")
    return value,pos_rec

def run_sar_backtest_v5(sar_agent, paradict, start_date: str, duration: int, df=pd.DataFrame([])):
    r = []
    date_list = []
    try:
        time_bars = df
        tmp_r = []
        value = pd.Series()
        pos_rec = pd.Series()
        last_val = 0
        last_pos = 0
        init_index = time_bars.index[0]
        c = 0
        last_high = 0
        for j in range(duration):
            timebar= time_bars[(init_index+(j)*datetime.timedelta(days=1)<=time_bars.index) & (time_bars.index<init_index+(j+1)*datetime.timedelta(days=1))]
            if not timebar.empty and timebar.shape[0] > 100:
                today_val, val, pos,today_pos,c,today_high= sar_stra_v5(timebar, paradict,last_high,sar_agent,last_pos,c)
                tmp_r.append((today_val-last_val) if (today_val-last_val) else 0)
                last_val = today_val
                last_pos = today_pos
                last_high = max(last_high,today_high)
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
    #print(f"每次胜率：{profit/(profit+loss)}")
    return value,pos_rec

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
        if self.strat == "v5":
            sar_stra_v5(time_bars_plot, self.paradict, agent_dict.get(self.agent),0,0,print_fig=True)

    def Backtest(self):
        agent_dict = {"v0": SAR_agent}
        if self.strat == "v3":
            return run_sar_backtest_v3(agent_dict[self.agent], self.paradict, self.start_time, self.duration, df=self.time_bars_all)
        if self.strat == "v4":
            return run_sar_backtest_v4(agent_dict[self.agent], self.paradict, self.start_time, self.duration, df=self.time_bars_all)
        if self.strat == "v5":
            return run_sar_backtest_v5(agent_dict[self.agent], self.paradict, self.start_time, self.duration, df=self.time_bars_all)
