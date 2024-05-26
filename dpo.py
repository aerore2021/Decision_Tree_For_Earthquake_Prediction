import pandas as pd
from category_encoders import OrdinalEncoder

def raw_clearing(raw):
    # 数据清洗，大概就把没有location的剔除掉了，别的数据没管
    raw = raw.drop(raw[raw['location'] == ''].index)
    raw = raw.reset_index(drop=True)
    return raw

def processing_time(data):
    # 将时间处理为数字：截取小时和分钟，剔除日期，以每天0点为基准，time=hours * 60 + minutes
    time = []
    for i in range(len(data)):
        time.append(int(data[i][12:13]) * 60 + int(data[i][15:16]))
    return pd.DataFrame(time)

def processing_discrete(data):
    # 离散数据编码，用target编码更好，但是懒
    oe = OrdinalEncoder()
    d = oe.fit_transform(data)
    return d
    
def construct_dataset(raw):
    raw = raw_clearing(raw)    
    # 把dataset设定成dataframe，图个方便
    dataset = pd.DataFrame()
    dataset_label = pd.DataFrame()
    '''
    # print(raw.columns)查看有什么种类的数据
    >>> Index(['title', 'magnitude', 'date_time', 'cdi', 'mmi', 'alert', 
            'tsunami', 'sig', 'net', 'nst', 'dmin', 'gap', 'magtype', 
            'depth', 'latitude', 'longitude', 'location', 'continent', 'country'])
    '''
    
    dataset["time"] = processing_time(raw['date_time'])
    # 震级，cdi，mmi和深度直接加
    dataset["magnitude"] = raw['magnitude']
    dataset["cdi"] = raw['cdi']
    dataset["mmi"] = raw['mmi']
    dataset["depth"] = raw['depth']
    dataset["latitude"] = raw['latitude']
    dataset["longitude"] = raw['longitude']
    dataset["location"] = processing_discrete(raw['location'])
    # dataset[""]
    dataset_label["tsunami"] = raw['tsunami']
    
    # 分析：magnitude,cdi,mmi,depth,latitude,longitude是连续值，location是离散值
    return dataset, dataset_label
