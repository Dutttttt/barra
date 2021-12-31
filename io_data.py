import pickle,sys,os
import pandas as pd,numpy as np
from pdb import set_trace 

def tackle_csv(df):
    "csv文件改变为数据库格式文件"
    df = df.applymap(np.float32)
    temp = df.unstack(1)
    temp.index = temp.index.swaplevel()
    temp.index = temp.index.rename(("TradingDay","SecuCode"))
    res = temp.reset_index();res["TradingDay"] = pd.to_datetime(res["TradingDay"])
    res = res.rename(columns = {0:"value"})
    res["SecuCode"] = res["SecuCode"].apply(lambda x:x.split('.')[0])
    return res

class io_d1():
    def __init__(self,p1 = None):
        if p1 is None:
            self._path = r"/Users/dt/Desktop/barra"
        else:
            self._path = p1
    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self,value):
        assert value["typ"] in ["full_name","part_name"],"typ argument error!"
        if value["typ"] == "full_name":
            self._path = value["value"]
        else:
            self._path = os.path.join(self._path,value["value"])
        print("私有属性path更新为:",self._path)
            
    @path.deleter
    def path(self):
        self._path = os.path.dirname(self._path)
        print("私有属性path更新为:",self._path)
        return self._path
        
    def load(self,name):
        with open(os.path.join(self._path,name),'rb') as f:
            data = pickle.load(f)
            return data
        
    def upload(self,value,name):
        with open(os.path.join(self._path,name),"wb") as f:
            pickle.dump(value,f)
            

if __name__ == "__main__":
    pass