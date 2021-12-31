import pandas as pd,numpy as np
import sys,os,datetime
from pdb import set_trace
from numpy.lib.stride_tricks import as_strided as stride
import statsmodels.api as sm
"""
研究区间2018-01-01-2021-01-01
将18年之后上市公司剔除
"""

class factor_handle:
    """
    收益率直接使用QT_Performance的ChangePCT项，不使用后复权系数计算
    原始因子值计算
    """
    def __init__(self):
        self.factor = {}
        
    @staticmethod
    def cut_time(df,time_col_name,start_time = pd.to_datetime("2016-01-01"),end_time = pd.to_datetime("2022-01-01")):
        use_col = df.columns.get_loc(time_col_name)
        return df[(df.iloc[:,use_col] < end_time)&(df.iloc[:,use_col] >= start_time)]
        
    @staticmethod
    def half_decrease(ret_series:np.ndarray,half_life=60):
        ret_series = ret_series.flatten()
        sita = np.power(0.5,1/half_life)
        coef = np.power(sita,np.arange(ret_series.shape[0]));
        coef = coef[::-1]/coef.sum()#权重归一化
        return ret_series*coef
    
    @staticmethod
    def one_beta_HSIGMA(ret_index:pd.DataFrame,ret_individual:pd.DataFrame):
        """
        如果交易日不满足250个，则设置为np.nan
        收益率序列250长度通过roll_beta会设置成250
        ret_index:columns->["TradingDay","ChangePCT"]
        ret_individual:columns->["TradingDay","ChangePCT"]
        """
        def one_day_beta(ret_index,ret_individual):
            ret_index_decrease = factor_handle.half_decrease(ret_index.values)
            ret_individual_decrease = factor_handle.half_decrease(ret_individual.values)
            Y = ret_index_decrease
            ret_individual_decrease = ret_individual_decrease[:,np.newaxis]
            X = sm.add_constant(ret_individual_decrease)
            model = sm.OLS(Y,X);results = model.fit()
            beta = results.params[-1]
            hsigma = np.std(results.resid)
            return pd.Series([beta,hsigma],index = ["beta","hsigma"])
        
        def roll_beta(df,length = 250):
            v = df.values;s0,s1 = v.shape;d0,d1 = v.strides
            if s0 - length + 1 > 0:
                res_ndarray = stride(v,(s0 - (length - 1),length,s1),(d0,d0,d1))
                rolled_df = pd.concat({row:pd.DataFrame(values,columns = df.columns)\
                                       for row,values in zip(df.index[length - 1:],res_ndarray)}) #时间不满足的不会生成
                res = rolled_df.groupby(level = 0).apply(lambda x:one_day_beta(x["hs300"],x["individual"]))
                return res
            else:
                return pd.DataFrame(np.nan*np.ones((df.shape[0],2)),index=df.index,columns = ["beta","hsigma"])
            
        #收益率半衰
        ret_index1 = ret_index.rename(columns = {"ChangePCT":"hs300"})
        ret_individual1 = ret_individual.rename(columns = {"ChangePCT":"individual"})
        ret_table = pd.merge(ret_index1.set_index("TradingDay"),ret_individual1.set_index("TradingDay"),left_index=True,right_index=True)
        res = roll_beta(ret_table)
        return res
    
    def beta_HSIGMA(self,ret_index:pd.DataFrame,ret_individual:pd.DataFrame):
        """
        ret_index:指数收益率table,TradingDay,SecuCode,ChangePCT
        ret_individual:个股收益table,TradingDay,SecuCode,ChangePCT
        columns均包含SecuCode,TradingDay,Price
        """
        beta = pd.DataFrame();hisgma = pd.DataFrame([])
        ret_index = factor_handle.cut_time(ret_index,"TradingDay")
        ret_individual = factor_handle.cut_time(ret_individual,"TradingDay")
        for code in pd.unique(ret_individual["SecuCode"]):
            print(datetime.datetime.now(),"-------",code,",start")
            individual_data = ret_individual[ret_individual["SecuCode"] == code];
            individual_data_temp = individual_data.drop(columns = ["SecuCode"])
            ret_index_temp = ret_index.drop(columns = ["SecuCode"])
            beta_one = factor_handle.one_beta_HSIGMA(ret_index_temp,individual_data_temp)
            beta_one["SecuCode"] = code;
            beta_one_temp = beta_one[["SecuCode","beta"]]
            hisgma_one_temp = beta_one[["SecuCode","hsigma"]]
            beta = pd.concat([beta,beta_one_temp],axis = 0)
            hisgma = pd.concat([hisgma,hisgma_one_temp],axis = 0)
        self.factor.update({"beta":beta})
        self.factor.update({"hisgma":hisgma})
        return beta,hisgma
        
    @staticmethod
    def last_half_decrease(ret_series:np.ndarray,last_day=21,half_life=120):
        sita = np.power(0.5,1/half_life)
        coef = np.power(sita,np.arange(ret_series.shape[0] - last_day))
        coef = np.r_[coef[::-1],np.zeros(last_day)]
        res = coef*ret_series;
        return res
    
    @staticmethod
    def one_momentum(df,all_day=500,last_day=21):
        """
        df的收益率ChangePCT要处以100
        df包含index:TradingDay,ChangePCT:Return
        """
        def one_day_momentum(df):
            ret_individual = df/100
            log_ret = np.log(ret_individual + 1)
            ret_individual_decrease = factor_handle.last_half_decrease(log_ret)
            return ret_individual_decrease.sum()
        return df.rolling(all_day+last_day).apply(one_day_momentum,raw = True)
        
    def momentum(self,df):
        """
        df:也只包括SecuCode,TradingDay,Return
        """
        momentum = pd.DataFrame([])
        df = factor_handle.cut_time(df,"TradingDay",start_time = pd.to_datetime("2015-01-01")).set_index("TradingDay")
        for code in pd.unique(df["SecuCode"]):
            print(datetime.datetime.now(),"-------",code,",start")
            temp_data = df[df["SecuCode"] == code]
            temp_data.drop(columns = ["SecuCode"],inplace = True)
            momentum_one = factor_handle.one_momentum(temp_data);momentum_one = pd.DataFrame(momentum_one);
            momentum_one["SecuCode"] = code
            momentum = pd.concat([momentum,momentum_one],axis = 0)
        self.factor.update({"momentum":momentum})
        return momentum
    
    @staticmethod
    def one_size(df):
        res = df.applymap(np.log)
        return res
    
    def size(self,df):
        """
        股票总市值取对数,columns:TradingDay,columns:TotalMV
        """
        size = pd.DataFrame([])
        df = factor_handle.cut_time(df,"TradingDay",start_time = pd.to_datetime("2018-01-01")).set_index("TradingDay")
        for code in pd.unique(df["SecuCode"]):
            temp_data = df[df["SecuCode"] == code]
            temp_data.drop(columns = ["SecuCode"],inplace=True)
            size_one = factor_handle.one_size(temp_data)
            size_one["SecuCode"] = code
            size = pd.concat([size,size_one],axis = 0)
        self.factor.update({"size":size})
        return size
    
    @staticmethod
    def one_EPIBS(est_eps,P):
        """
        个股一致预期基本每股收益/当前股票价格
        """
        res = est_eps/P
        return res
    
    def EPIBS(self,est_eps_df,P_df):
        """
        est_eps_df:TradingDay,SecuCode,value
        P_df:TradingDay,SecuCode,value
        """
#         set_trace()
        est_eps_df = factor_handle.cut_time(est_eps_df,"TradingDay").set_index("TradingDay")
        P_df = factor_handle.cut_time(P_df,"TradingDay").set_index("TradingDay")
        EPIBS = pd.DataFrame([])
        for code in pd.unique(est_eps_df["SecuCode"]):
            temp_est_eps = est_eps_df[est_eps_df["SecuCode"] == code]
            temp_est_eps.drop(columns = ["SecuCode"],inplace = True)
            temp_P = P_df[P_df["SecuCode"] == code]
            temp_P.drop(columns = ["SecuCode"],inplace = True)
            EPIBS_one = pd.DataFrame(factor_handle.one_EPIBS(temp_est_eps,temp_P))
            EPIBS_one["SecuCode"] = code
            EPIBS = pd.concat([EPIBS,EPIBS_one],axis = 0)
        self.factor.update({"EPIBS":EPIBS})
        return EPIBS
    
    def ETOP(self,pe):
        "过去12月个股净利润/当前市值"
        res = 1/(pe.set_index(["TradingDay","SecuCode"]))
        self.factor.update({"ETOP":res.reset_index()})
        return res
    
    @staticmethod
    def one_CETOP(cash_earnings,P):
        """
        cash_earnings:TradingDay,SecuCode,value
        P:TradingDay,SecuCode,value
        """
        res = cash_earnings/P
        return res
        
    def CETOP(self,cash_earnings,P):
        """
        cash_earnings和P的输入值要重新命名列名为value
        """
        cash_earnings = factor_handle.cut_time(cash_earnings,"EndDate").set_index("EndDate");
        P = factor_handle.cut_time(P,"TradingDay").set_index("TradingDay")
        res = pd.DataFrame([])
        for code in pd.unique(P["SecuCode"]):
            cash_earnings_temp = cash_earnings[cash_earnings["SecuCode"] == code]
            P_temp = P[P["SecuCode"] == code]
            cash_earnings_temp.drop(columns = ["SecuCode"],inplace = True)
            cash_earnings_temp = cash_earnings_temp.rename(columns = {"cash_profit":"value"}).sort_index()
            P_temp.drop(columns = ["SecuCode"],inplace = True)
            P_temp = P_temp.rename(columns = {"ClosePrice":"value"}).sort_index()
            cash_earnings_temp = cash_earnings_temp.reindex(P_temp.index).ffill()#变更频率
            CETOP_one = factor_handle.one_CETOP(cash_earnings_temp,P_temp)
            CETOP_one["SecuCode"] = code
            res = pd.concat([res,CETOP_one],axis = 0)
        self.factor.update({"CETOP":res})
        return res
    
    def DASTD(self,ret_df,rolling_num=250):
        """
        输入ret_df: columns:SecuCode, TradingDay, ChangePCT
        """
        def one_DASTD(ret_series):
            ret_series = ret_series.flatten()
            ret_series_temp = (ret_series - ret_series.mean())**2
            res = factor_handle.half_decrease(ret_series_temp,half_life = 40)
            return np.sqrt(res.sum())
        DASTD = pd.DataFrame([])
        ret_df = factor_handle.cut_time(ret_df,"TradingDay").set_index("TradingDay");
        for code in pd.unique(ret_df["SecuCode"]):
            ret_df_temp = ret_df[ret_df["SecuCode"] == code];ret_df_temp = ret_df_temp.drop(columns = ["SecuCode"])
            DASTD_temp = ret_df_temp.rolling(rolling_num).apply(one_DASTD,raw=True)
            DASTD_temp = pd.DataFrame(DASTD_temp);DASTD_temp["SecuCode"] = code;
            DASTD = pd.concat([DASTD,DASTD_temp],axis = 0)
        self.factor.update({"DASTD":DASTD})
        return DASTD
    
    @staticmethod
    def one_CMRA(ret_month):
        def temp_CMRA(ret_month:np.ndarray):
            log_ret_month = np.log(ret_month + 1)
            max_log_ret = max(log_ret_month);min_log_ret = min(log_ret_month)
            return np.log(1+max_log_ret) - np.log(1+min_log_ret)
        res = ret_month.rolling(12).apply(temp_CMRA,raw=True)
        return res
    
    def CMRA(self,ret_df):
        """
        ret_df要变更为月收益率
        ret_df:columns:TradingDay,SecuCode,ChangePCT
        """
        CMRA = pd.DataFrame([])
        ret_df = factor_handle.cut_time(ret_df,"TradingDay").set_index("TradingDay")
        for code in pd.unique(ret_df["SecuCode"]):
            temp_ret = ret_df[ret_df["SecuCode"] == code];temp_ret.drop(columns = ["SecuCode"],inplace = True)
            #变为月频率
            temp_ret1 = temp_ret.resample("M").sum()
            part_CMRA = factor_handle.one_CMRA(temp_ret1);part_CMRA = pd.DataFrame(part_CMRA);
            part_CMRA["SecuCode"] = code
            CMRA = pd.concat([CMRA,part_CMRA],axis = 0)
        self.factor.update({"CMRA":CMRA})
        return CMRA
    
    @staticmethod
    def one_SGRO(Operating_table,rolling_year = 5):
        """
        Operating_table:TradingDay,SecuCode,value
        之前先要把EndDate修改列名为TradingDay
        """
        rolling_num = rolling_year*4;
        res = Operating_table.set_index("TradingDay")["value"]\
        .rolling(rolling_num).apply(lambda x:np.power(x.iloc[-1]/x.iloc[0],1/5) - 1).dropna() #以000001为例，应该以2016-12-31为起点进行计算
        return res
    
    def SGRO(self,Operating_all_table,rolling_year = 5):
        """
        以000001的时间轴进行对齐，空缺处填nan
        此处不用cut_time
        Operating_all_table:columns->TradingDay,SecuCode,value
        """
        SGRO = pd.DataFrame([])
        for code in sorted(pd.unique(Operating_all_table["SecuCode"])):
            temp_Operating_table = Operating_all_table[Operating_all_table["SecuCode"] == code]
            temp_Operating_table = temp_Operating_table.sort_values("TradingDay")
            part_SGRO = factor_handle.one_SGRO(temp_Operating_table);
            part_SGRO = pd.DataFrame(part_SGRO);
            if code == "000001":
                valid_index = part_SGRO.index
            else:
                part_SGRO = part_SGRO.reindex(valid_index)
            part_SGRO["SecuCode"] = code
            SGRO = pd.concat([SGRO,part_SGRO],axis = 0)
        self.factor.update({"SGRO":SGRO})
        return SGRO
    
    @staticmethod 
    def one_EGRO(Mu_profit_table,rolling_year = 5):
        """
        Mu_profit_table:TradingDay,SecuCode,value
        之前先要把EndDate修改列名为TradingDay
        """
        rolling_num = rolling_year*4;
        res = Mu_profit_table.set_index("TradingDay")["value"]\
        .rolling(rolling_num).apply(lambda x:np.power(x.iloc[-1]/x.iloc[0],1/5) - 1).dropna() 
        return res
    
    def EGRO(self,Mu_profit_all_table,rolling_year = 5):
        """
        Mu_profit_all_table:TradingDay,SecuCode,value
        """
        EGRO = pd.DataFrame([])
        for code in sorted(pd.unique(Mu_profit_all_table["SecuCode"])):
            temp_Mu_profit = Mu_profit_all_table[Mu_profit_all_table["SecuCode"] == code].sort_values("TradingDay")
            part_EGRO = factor_handle.one_EGRO(temp_Mu_profit);
            part_EGRO = pd.DataFrame(part_EGRO);
            if code == "000001":
                valid_index = part_EGRO.index
            else:
                part_EGRO = part_EGRO.reindex(valid_index)
            part_EGRO["SecuCode"] = code
            EGRO = pd.concat([EGRO,part_EGRO],axis = 0)
        self.factor.update({"EGRO":EGRO})
        return EGRO
    
    def EGIB(self,EGIB_all):
        self.factor.update({"EGIB":EGIB_all})
        return EGIB_all
    
    def EGIB_S(self,EGIB_S_all):
        self.factor.update({"EGIB_s":EGIB_S_all})
        return EGIB_S_all
    
    @staticmethod
    def one_BTOP(common_equity,market_capital):
        """
        commo_equity:TradingDay,value
        market_capital:TradingDay,value
        """
        return common_equity/market_capital
    
    def BTOP(self,common_equity_table,market_capital_table):
        """
        commo_equity:TradingDay(pd.to_datetime),SecuCode,value
        market_capital:TradingDay(pd.to_datetime),SecuCode,value
        """
        BTOP = pd.DataFrame([])
        for code in sorted(pd.unique(common_equity_table["SecuCode"])):
            temp_common_equity = common_equity_table[common_equity_table["SecuCode"] == code].sort_values("TradingDay").\
            set_index("TradingDay")
            temp_market_capital_table = market_capital_table[market_capital_table["SecuCode"] == code].sort_values("TradingDay").set_index("TradingDay")
            temp_common_equity.drop(columns = ["SecuCode"],inplace=True)
            temp_market_capital_table.drop(columns = ["SecuCode"],inplace=True)
            temp_common_equity = temp_common_equity.reindex(index = temp_market_capital_table.index).ffill()
            temp_BTOP = factor_handle.one_BTOP(temp_common_equity,temp_market_capital_table)
            temp_BTOP["SecuCode"] = code
            BTOP = pd.concat([BTOP,temp_BTOP],axis = 0)
        self.factor.update({"BTOP":BTOP})
        return BTOP
    
    @staticmethod
    def one_MLEV(ME,LD):
        """
        ME:企业当前总市值,TradingDay,SecuCode,value
        LD:企业长期负债,TradingDay,SecuCode,value
        """
        return ((ME + LD)/ME).ffill()
    
    def MLEV(self,ME_table,LD_table):
        """
        ME总市值
        LD长期负债
        """
        MLEV = pd.DataFrame([])
        for code in sorted(pd.unique(ME_table["SecuCode"])):
            temp_ME_table = ME_table[ME_table["SecuCode"] == code].set_index("TradingDay")
            temp_ME_table.drop(columns = "SecuCode",inplace=True)
            temp_LD_table = LD_table[LD_table["SecuCode"] == code].sort_values("TradingDay").set_index("TradingDay")
            temp_LD_table.drop(columns = "SecuCode",inplace=True)
            temp_LD_table = temp_LD_table.reindex(temp_ME_table.index).ffill()
            temp_MLEV = factor_handle.one_MLEV(ME = temp_ME_table,LD = temp_LD_table)
            temp_MLEV["SecuCode"] = code
            MLEV = pd.concat([MLEV,temp_MLEV],axis=0)
        self.factor.update({"MLEV":MLEV})
        return MLEV
    
    def DTOA(self,TD_TA_table):
        """
        TD:总负债,TradingDay,SecuCode,value
        TA:总资产,TradingDay,SecuCode,value
        """
        self.factor.update({"DTOA":TD_TA_table})
        return TD_TA_table
        
    @staticmethod
    def one_BLEV(BE,LD):
        return ((BE + LD)/BE).ffill()
    
    def BLEV(self,BE_table,LD_table):
        """
        BE:企业账面权益
        LD:企业长期负债
        """
        BLEV = pd.DataFrame([])
        for code in sorted(pd.unique(BE_table["SecuCode"])):
            temp_BE_table = BE_table[BE_table["SecuCode"] == code]
            temp_BE_table = factor_handle.cut_time(temp_BE_table,"TradingDay").set_index("TradingDay").sort_index()
            temp_LD_table = LD_table[LD_table["SecuCode"] == code]
            temp_LD_table = factor_handle.cut_time(temp_LD_table,"TradingDay").set_index("TradingDay").sort_index()
            temp_BE_table.drop(columns = ["SecuCode"],inplace=True)
            temp_LD_table.drop(columns = ["SecuCode"],inplace=True)
            temp_BLEV = factor_handle.one_BLEV(temp_BE_table,temp_LD_table)
            temp_BLEV["SecuCode"] = code
            BLEV = pd.concat([BLEV,temp_BLEV],axis=0)
        self.factor.update({"BLEV":BLEV})
        return BLEV
    
    @staticmethod
    def one_STOM(df):
        """
        df,columns:SecuCode,Volume,Stock_equity
        df,index:TradingDay
        Volume:当日成交量
        Stock_equity:流通股本
        
        return:TradingDay,SecuCode,STOM
        """
        def roll(df,length = 21):
            df = df.dropna();df = df.set_index("TradingDay")
            v = df.values;s0,s1 = v.shape;d0,d1 = v.strides;
            if s0 - length + 1 > 0:
                res_ndarray = stride(v,(s0 - (length - 1),length,s1),(d0,d0,d1))
                rolled_df = pd.concat({row:pd.DataFrame(values,columns = df.columns) \
                                     for row,values in zip(df.index[length - 1:],res_ndarray)})
                def func(df):
                    Vt = df["Volume"].values.flatten();St = df["Stock_equity"].values.flatten()
                    return pd.Series(np.log(Vt/St).sum(),index = ["STOM"])
                res = rolled_df.groupby(level = 0).apply(lambda x:func(x))
                return res.reset_index()
            else:
                import copy
                temp = copy.copy(df.index)
                temp.name = "index"
                res = pd.DataFrame(np.nan*np.ones((df.shape[0],1)),index=temp,columns = ["STOM"])
                return res.reset_index()
        return roll(df)
        
    def STOM(self,df):
        """
        df,columns:TradingDay,SecuCode,Volume,Stock_equity
        
        return:TradingDay,SecuCode,STOM
        """
        STOM = pd.DataFrame([])
        for code in sorted(pd.unique(df["SecuCode"])):
            temp_code_df = df[df["SecuCode"] == code];temp_code_df.drop(columns=["SecuCode"],inplace=True)
            temp_STOM = factor_handle.one_STOM(temp_code_df);temp_STOM["SecuCode"] = code
            STOM = pd.concat([STOM,temp_STOM],axis = 0)
        self.factor.update({"STOM":STOM})
        return STOM
    
    @staticmethod
    def one_STOQ(df,T = 3):
        """
        df,columns:TradingDay,SecuCode,STOM
        return:TradingDay,SecuCode,STOQ
        """
        code = df.iloc[0,df.columns.get_loc("SecuCode")]
        res = df.set_index("TradingDay")["STOM"].rolling(T).apply(lambda x:np.log(np.exp(x).mean()))
        res.name = "STOQ";res = pd.DataFrame(res);
        res["SecuCode"] = code
        return res.reset_index()
    
    def STOQ(self,df):
        """
        df,columns:TradingDay,SecuCode,STOM
        
        return:TradingDay,SecuCode,STOQ
        """
        STOQ = pd.DataFrame([])
        for code in sorted(pd.unique(df["SecuCode"])):
            temp_code_df = df[df["SecuCode"] == code]
            temp_STOQ = factor_handle.one_STOQ(temp_code_df)
            STOQ = pd.concat([STOQ,temp_STOQ],axis = 0)
        self.factor.update({"STOQ":STOQ})
        return STOQ
    
    @staticmethod
    def one_STOA(df,T = 12):
        """
        df,columns:TradingDay,SecuCode,STOA
        
        return:TradingDay,SecuCode,STOA
        """
        code = df.iloc[0,df.columns.get_loc("SecuCode")]
        res = df.set_index("TradingDay")["STOM"].rolling(T).apply(lambda x:np.log(np.exp(x).mean()))
        res.name = "STOA";res = pd.DataFrame(res);
        res["SecuCode"] = code
        return res.reset_index()
    
    def STOA(self,df):
        STOA = pd.DataFrame([])
        for code in sorted(pd.unique(df["SecuCode"])):
            temp_code_df = df[df["SecuCode"] == code]
            temp_STOA = factor_handle.one_STOA(temp_code_df)
            STOA = pd.concat([STOA,temp_STOA],axis = 0)
        self.factor.update({"STOA":STOA})
        return STOA
    
if __name__ == "__main__":
    pass