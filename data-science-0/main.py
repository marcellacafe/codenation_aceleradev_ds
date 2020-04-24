#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[4]:

import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


type(black_friday)


# In[5]:


black_friday.head(10)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[8]:


def q1():
    # Retorne aqui o resultado da questão 1.
    result = black_friday.shape
    return result


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[9]:


def q2():
    # Retorne aqui o resultado da questão 2.
    qty_woman = len(black_friday.loc[(black_friday["Age"] == "26-35") & (black_friday["Gender"] == "F")])
    return qty_woman


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    unique_user = black_friday['User_ID'].nunique()
    return unique_user


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    n_types = black_friday.dtypes.nunique()
    return n_types


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    # perc = (total-na)/total
    perc_na = (len(black_friday) - len(black_friday.dropna())) / len(black_friday)
    return perc_na


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[1]:


def q6():
    # Retorne aqui o resultado da questão 6.
    # Soma os valores NULL e depois seleciono o maior valor
    # É necessário converter o resultado em int para passsar no teste
    bigger_null = (black_friday.isnull().sum()).max()
    
    return int(bigger_null)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    # Para achar o número mais frequente basta calcular a moda
    num_freq = float(black_friday['Product_Category_3'].mode())
    return num_freq


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    # Retorne aqui o resultado da questão 8.
    purchase_min = black_friday['Purchase'].min()
    purchase_max = black_friday['Purchase'].max()
    purchase_mean = ((black_friday['Purchase'] - purchase_min) / (purchase_max - purchase_min)).mean() 
    return float(purchase_mean)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    # Retorne aqui o resultado da questão 9.
    purchase_padronizado = (black_friday['Purchase'] - np.mean(black_friday['Purchase'])) / np.std(black_friday['Purchase'])
    return len([i for i in purchase_padronizado if i > -1 and i < 1])


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    # Retorne aqui o resultado da questão 10.
    check = (black_friday['Product_Category_2'].isna() == black_friday['Product_Category_2'].isna()).all()
    return bool(check)

