import pandas as pd,numpy as np
import sys,os
from pdb import set_trace
import importlib
"""
采用行业均值填充缺失值
"""

def find_null(data,code_all,value_name):
    """
    找寻有缺失的证券代码
    """
    res = []
    for code in code_all:
        if pd.isnull(data.loc[data["SecuCode"] == code,value_name]).any():
            res.append(code)
    return res

def get_similar_industry(industry_info,use_data,code,col_name):
    """
    获得主板非创业板同行业
    """
    target_industry = industry_info.loc[industry_info["code"] == code,"index_code"].iloc[0]
    whole_individual = list(industry_info.loc[industry_info["index_code"] == target_industry,"code"].values.flatten())
    _ = whole_individual.pop(whole_individual.index(code))
    whole_data = use_data[pd.Index(use_data["SecuCode"]).isin(whole_individual)]
    index_name = "index" if use_data.index.name is None else use_data.index.name
    res = pd.DataFrame(whole_data.reset_index().pivot(index=index_name,columns = "SecuCode",values=col_name).mean(1,skipna=True))
    res = res.rename(columns = {0:col_name});res.index.name = None
    res = res.reindex(columns = use_data.columns)
    res["SecuCode"] = code
    return res

def fill_null(industry_info,use_data,code,col_name):
    t1 = get_similar_industry(industry_info,use_data,code,col_name)
    nan_data = use_data.loc[use_data["SecuCode"] == code]
    temp_data = nan_data[pd.isnull(nan_data[col_name])]
    start_time = temp_data.index[0];end_time = temp_data.index[-1]
    nan_data.loc[start_time:end_time,col_name] = t1.loc[start_time:end_time,col_name]
    use_data.loc[use_data["SecuCode"] == code] = nan_data
    return use_data

def fill_null_codelist(industry_info,use_data,code_list,col_name):
    for code in code_list:
        use_data = fill_null(industry_info,use_data,code,col_name)
    return use_data

def fill_with_market(data,col_name):
    null_code = pd.unique(data[pd.isnull(data[col_name])]["SecuCode"])
    valid_data = data[pd.notnull(data[col_name])].reset_index().pivot(index = "TradingDay",columns = "SecuCode",values = col_name)
    valid_data_mean = valid_data.mean(1,skipna=True)
    for code in null_code:
        null_code_data = data[data["SecuCode"] == code]
        nan_data = null_code_data[pd.isnull(null_code_data[col_name])]
        start_time = nan_data.index[0];end_time = nan_data.index[-1]
        null_code_data.loc[start_time:end_time,col_name] = valid_data_mean.loc[start_time:end_time,col_name]
        data.loc[data["SecuCode"] == code] = null_code_data
    return data

if __name__ == "__main__":
    pass
