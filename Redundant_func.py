import pandas as pd,numpy as np
import sys,os,datetime
from pdb import set_trace

def extract_valid_company(hs_code,zz_code,typ = "valid"):
    """
    获取有效公司名
    """
    assert typ in ["valid","kechuang"],"argument error!"
    all_code = pd.Index(hs_code).union(pd.Index(zz_code))
    if typ == "valid":
        return all_code[all_code < "689000"]
    elif typ == "kechuang":
        return all_code[all_code < "688000"]
    
def extract_valid_code(data,typ = "kechuang"):
    if typ == "valid":
        return data[data < "689000"]
    elif typ == "kechuang":
        return data[data < "688000"]
    
def component_select_code(df,start_time:datetime.datetime,end_time:datetime.datetime):
    """
    从HS_comp或者ZZ_comp中筛选动态成分股
    OutDate在start_time之后或者为空值
    InDate在end_time之前
    """
    res = []
    for ind,val in df.iterrows():
        Indate = val["InDate"];Outdate = val["OutDate"];
        if (Indate < end_time) & ((Outdate > start_time) | (pd.isnull(Outdate))):
            res.append(val["SecuCode"])
    return np.asarray(res)

def get_valid_zb_code(zb_code,stock_pool):
    res = np.asarray([code for code in stock_pool if code in zb_code])
    return res
    
if __name__ == "__main__":
    pass