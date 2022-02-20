#!/usr/bin/env python
# coding: utf-8

# # H&M Personalized Fashion Recommendations
# Provide product recommendations based on previous purchases

# ## 1. EDA
# - articles
# - customers
# - transactions_train
# - images

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# In[6]:


def unique(x):
    import numpy as np
    x = np.array(x)
    return np.unique(x)


# In[43]:


art = pd.read_csv('articles.csv')
print('[articles]')
print('Articles data with %d rows and %d columns.' % (art.shape[0], art.shape[1]))
print('\nColumns in articles:\n', art.columns.values)


# In[44]:


art.head()


# 
# |**變數名稱**|**變數介紹**|
# |:-|:-|
# |article_id|商品id(key)|
# |product_code, prod_name|商品的編碼及名稱(相同的編碼或名稱下有不一樣的特徵就屬於不同的產品)|
# |product_type_no, product_type_name|商品類型的編號、名稱|
# |product_group_name|商品分類名稱|
# |graphical_appearance_no, graphical_appearance_name|圖形外觀的編號、名稱(one-to-one)|
# |colour_group_code, colour_group_name|顏色的編號、名稱(one-to-one)|
# |perceived_colour_value_id, perceived_colour_value_name|明亮度及色調的編碼、名稱|
# |perceived_colour_master_id, perceived_colour_master_name|主顏色的編號、名稱|
# |department_no, department_name|department編碼、名稱|
# |index_code, index_name|index編碼、名稱|
# |index_group_no, index_group_name|index種類編碼、名稱|
# |section_no, section_name|section編碼、名稱|
# |garment_group_no, garment_group_name|服裝類別的編號及名稱|
# |detail_desc|商品敘述|

# In[45]:


# check missing values
print('Check missing values:')
art.isnull().sum(axis = 0)


# In[5]:


# unique values in all columns
for i in art.columns[:24]:
    print('number of unique values in ', i, ' : ', len(unique(art[i])), sep = '')

unique_desc = []
for i in art['detail_desc']:
    if i not in unique_desc:
        unique_desc.append(i)
print('number of unique values in detail_desc:', len(unique_desc))

print('')

for i in art.columns[:23]:
    print('unique values in ', i, ' -> ', unique(art[i]), sep = '')

print('unique values in garment_group_name -> ', unique(art['garment_group_name']), sep = '')


# In[8]:


print('unique values in detail_desc ->')
unique_desc


# articles.csv中包含了商品的資訊，共有105,542個商品及25個變數，其中在detail_desc中有416個空值，即沒有商品介紹。另外根據上方的唯一值資訊以及變數之間的關係可以做些猜測，例如從資料可看出，一些變數之間呈現1對1的關係，例如product_type_no與product_type_name、graphical_appearance_no與graphical_appearance_name及colour_group_code與colour_group_name等變數，這些都可以作為未來決定要放入那些特徵進入模型的考量。

# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


trans = pd.read_csv('transactions_train.csv')

display(trans.head())
trans.shape


# In[ ]:





# In[ ]:





# In[ ]:




