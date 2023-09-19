# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:47:26 2021

@author: karthikeyan
"""

import numpy as np
import pandas as pd
import glob
import os
import re
import math
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import kaleido
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from scipy import stats, optimize, interpolate
from scipy.stats import zscore


raw_tax_path = 'data/raw_data/tax_csv2/'
raw_census_path = 'data/raw_data/census_csv/'
raw_sgu_path = 'data/raw_data/sgu_solar_csv/'
wrangled_data_path = 'data/wrangled_data/'
plots_path = 'plots/'
raw_geo_path = 'data/raw_data/aust_shape/'

# Changes this to True if you have Geo Pandas installed. If you can't seem to make it work on th python console,
# please use the IPython Notebook that upload in the GitHub. However, If you want to run the codes, you might have to
#download the geopandas int your anaconda to make it work. Otherwise, feel free to use the pre-run notebook to verify.

IS_GEOPANDAS = False

def create_folder(path): 
    if not(os.path.exists(path)):    
        os.makedirs(path)

create_folder(wrangled_data_path)
create_folder(plots_path)

# 1. Extrapolate Viable bulidings and Total buildings from 2001 - 2018 based on increase rate between 2006, 2011 and 2011, 2016

def create_buildings_data():
    #  1) Fetch data from Census website and create a dataframe of viable buildings in 2011 and 2016
    Data_2006 = pd.read_csv(raw_census_path + 'Census2006New.csv',encoding = 'utf-8-sig')
    Data_2011 = pd.read_csv(raw_census_path + 'Census2011New.csv',encoding = 'utf-8-sig')
    Data_2016 = pd.read_csv(raw_census_path + 'Census2016New.csv',encoding = 'utf-8-sig')
    
    #Data_2006 = pd.read_csv(raw_census_path + 'Census2006New.csv',encoding = 'ISO-8859-1')
    #Data_2011 = pd.read_csv(raw_census_path + 'Census2011New.csv',encoding = 'ISO-8859-1')
    #Data_2016 = pd.read_csv(raw_census_path + 'Census2016New.csv',encoding = 'ISO-8859-1')

    #  2) Process 2006 data
    separate_2006 = Data_2006['Separate house']
    semi_2006 = Data_2006['Semi-detached, row or terrace house, townhouse etc with one storey']+Data_2006['Semi-detached, row or terrace house, townhouse etc with two or more storeys']
    region_id = Data_2006['Postcode'].astype(str)
    total_buildings_2006 = Data_2006['Total']
    total_2006 = semi_2006 + separate_2006
    viable_2006 = pd.DataFrame({'region_id':region_id, 'viable_buildings_2006': total_2006, 'total_buildings_2006': total_buildings_2006}) 
    viable_2006.set_index('region_id')
    # - Convert float to int value
    viable_2006['viable_buildings_2006'] = np.floor(pd.to_numeric(viable_2006['viable_buildings_2006'], errors='coerce')).astype('Int64')
    viable_2006['total_buildings_2006'] = np.floor(pd.to_numeric(viable_2006['total_buildings_2006'], errors='coerce')).astype('Int64')

    #  3) Process 2011 data
    separate_2011 = Data_2011['Separate house']
    semi_2011 = Data_2011['Semi-detached, row or terrace house, townhouse etc with one storey'] + Data_2011['Semi-detached, row or terrace house, townhouse etc with two or more storeys']
    region_id = Data_2011['Postcode'].astype(str).str[:4]
    total_buildings_2011 = Data_2011['Total']
    total_2011 = semi_2011 + separate_2011
    viable_2011 = pd.DataFrame({'region_id':region_id, 'viable_buildings_2011': total_2011, 'total_buildings_2011': total_buildings_2011}) 
    viable_2011.set_index('region_id')
    #  - Convert float to int value
    viable_2011['viable_buildings_2011'] = np.floor(pd.to_numeric(viable_2011['viable_buildings_2011'], errors='coerce')).astype('Int64')
    viable_2011['total_buildings_2011'] = np.floor(pd.to_numeric(viable_2011['total_buildings_2011'], errors='coerce')).astype('Int64')

    #  4) Process 2011 data
    separate_2016 = Data_2016['Separate house']
    semi_2016 = Data_2016['Semi-detached, row or terrace house, townhouse etc. with one storey'] + Data_2016['Semi-detached, row or terrace house, townhouse etc. with two or more storeys']
    region_id = Data_2016['Postcode'].astype(str).str[:4]
    total_buildings_2016 = Data_2016['Total']
    total_2016 = semi_2016 + separate_2016
    viable_2016 = pd.DataFrame({'region_id':region_id, 'viable_buildings_2016': total_2016, 'total_buildings_2016': total_buildings_2016}) 
    viable_2016.set_index('region_id')
    #  - Convert float to int value
    viable_2016['viable_buildings_2016'] = np.floor(pd.to_numeric(viable_2016['viable_buildings_2016'], errors='coerce')).astype('Int64')
    viable_2016['total_buildings_2016'] = np.floor(pd.to_numeric(viable_2016['total_buildings_2016'], errors='coerce')).astype('Int64')

    #  5) Merge three dataframes
    two = pd.merge(viable_2006, viable_2011, on="region_id", how='inner')
    all_buildings_data = pd.merge(two, viable_2016, on="region_id", how='inner')
    all_buildings_data.sort_values(['region_id'], inplace=True)


    #  6) Predict value of viable buildings and total buildings based on y=ax+b
    region_id = all_buildings_data['region_id']
    new = pd.DataFrame({'region_id':region_id})
    t_new = pd.DataFrame({'region_id':region_id})

    def fin_value(a, b, year):
        result =((year*a) + b)
        if result < 0 :
            return 0
        return result

    for i in range(1,11):
        v_lst = []
        t_lst = []
        str_name = str(2000+i)
        for index, row in all_buildings_data.iterrows():
            a1 = (row['viable_buildings_2011']- row['viable_buildings_2006'])/(5)
            b1 = row['viable_buildings_2011'] - (a1 * 11)
            a2 = (row['total_buildings_2011']- row['total_buildings_2006'])/(5)
            b2 = row['total_buildings_2011'] - (a2 * 11)
            v_result = int(np.round(fin_value(a1,b1,i)))
            t_result = int(np.round(fin_value(a2,b2,i)))
            v_lst.append(v_result)
            t_lst.append(t_result)
        new.insert(i, str_name, v_lst, True)
        t_new.insert(i, str_name, t_lst, True)

    for i in range(11,20):
        v_lst = []
        t_lst = []
        str_name = str(2000+i)
        for index, row in all_buildings_data.iterrows():
            a1 = (row['viable_buildings_2016']- row['viable_buildings_2011'])/(5)
            b1 = row['viable_buildings_2016'] - (a1 * 16)
            a2 = (row['total_buildings_2016']- row['total_buildings_2011'])/(5)
            b2 = row['total_buildings_2016'] - (a2 * 16)
            v_result = int(np.round(fin_value(a1,b1,i)))
            t_result = int(np.round(fin_value(a2,b2,i)))
            v_lst.append(v_result)
            t_lst.append(t_result)
        new.insert(i, str_name, v_lst, True)
        t_new.insert(i, str_name, t_lst, True)

    new.to_csv(wrangled_data_path + 'total_viable_buildings.csv', index=False)
    t_new.to_csv(wrangled_data_path + 'total_buildings.csv', index=False)


def create_wrangled_data():
    # set up empty dataframes to start collecting data
    num_people = pd.DataFrame(index=['3000'], columns=['2001'])
    num_people.index.name='postcode'
    total_income = pd.DataFrame(index=['3000'], columns=['2001'])
    total_income.index.name='postcode'
    
    
    # compile number individuals and taxable income data, income data layout  is variable
    # from year to year
    for filename in glob.glob(raw_tax_path + '*table*.csv'):
        print(filename)
        year = int(re.findall(r'[\d]{4}', filename)[0])
        # note postcodes switch to second column from 2012 onwards (except 2013)
        # from 2018 onwards column headings in second row
        if year <= 2011 or year == 2013:
            get_table = pd.read_csv(filename, header=2, index_col=1, dtype=str)
        elif year <= 2017:
            get_table = pd.read_csv(filename, header=2, index_col=2, dtype=str)
        else:
            get_table = pd.read_csv(filename, header=1, index_col=2, dtype=str)
        get_table = get_table[get_table.index.notnull()]
        get_table.columns = map(str.lower, get_table.columns)
        get_table.index = get_table.index.map(str)
        
        # split out number individuals and taxable income data
        # note that the column headings and layout changed at various points
        # until 2009 need to sum non-taxable and taxable individulas to get
        # num individauls
        # where > 1 "taxable income" column is pulled back the maximum is
        # the relevant one
        if year <= 2009:
            ppl_table = get_table[list(col for col in get_table.columns if 
                                       'taxable individuals' in col)].copy().astype(float)
            ppl_table['use_num']=ppl_table.sum(axis=1)
            inc_table = get_table[list(col for col in get_table.columns if 
                                       'taxable income' in col)].copy().astype(float)
            inc_table['use_income']=inc_table.max(axis=1)
        else:
            ppl_table = get_table[list(col for col in get_table.columns if 
                                       'number of individuals' in col)].copy().astype(float)
            ppl_table['use_num']=ppl_table.max(axis=1)
            inc_table = get_table[list(col for col in get_table.columns if 
                                       'taxable income or loss' in col)].copy().astype(float)
            inc_table['use_income']=inc_table.max(axis=1)   
        # if a new year (ie other than 2001) then add a column
        if not(str(year) in num_people.columns):
            num_people[str(year)]=np.NaN
            total_income[str(year)]=np.NaN
        
        for postcode in inc_table.index:
            if postcode.isnumeric():
                # only want VIC data
                if postcode >= '3000' and postcode < '4000':            
                    # if postcode not in tables, then need to add
                    if not(postcode in num_people.index):
                        num_people = num_people.append(pd.DataFrame(index=[postcode]))
                        total_income = total_income.append(pd.DataFrame(index=[postcode]))                    
                    num_people.loc[postcode, str(year)] = ppl_table.loc[postcode,'use_num']
                    total_income.loc[postcode, str(year)] = inc_table.loc[postcode,'use_income']    
     
    # save down the compiled tax data
    num_people.to_csv(wrangled_data_path + 'num_people.csv')
    total_income.to_csv(wrangled_data_path + 'total_income.csv')
    
    # collate financial year installation data (files are by calendar year)
    install_data = num_people.copy()
    install_num = num_people.copy()
    for col in install_data.columns:
        install_data[col].values[:] = 0
        install_num[col].values[:] = 0
        
    # these are used to break data into correct financial year
    first_half_months = ['Jan','Feb','Mar','Apr','May','Jun']
    
    for filename in glob.glob(raw_sgu_path + '*SGU-Solar*.csv'):    
        year = int(re.findall(r'[\d]{4}', filename)[0])    
        # only pull back even years, as 2 years of data in each file
        if year % 2 == 0 or year == 2019:
            get_table = pd.read_csv(filename, dtype=str)
            print(filename)
            # first data set starts in apr, not jan
            # 2019 only fetch one year of data
            # all other years take in two years of data
            if year == 2002:
                get_cols = [number for number in range(8,43) if number % 2 == 0]        
            elif year == 2019:
                get_cols = [number for number in range(28,51) if number % 2 == 0]        
            else:
                get_cols = [number for number in range(4,51) if number % 2 == 0]        
            
            for row in range(len(get_table)):
                get_pcode = int(get_table.iloc[row,0])
                if get_pcode >=3000 and get_pcode <4000 and (str(get_pcode) in install_data.index):
                    for col in get_cols:
                        get_year = int(get_table.columns[col][4:8])
                        get_month = get_table.columns[col][0:3]
                        # if in first six months of year allocate to prior year
                        if get_month in first_half_months:
                            get_year -= 1
                        if get_year <= 2018:
                            install_data.loc[str(get_pcode), str(get_year)] += float(re.sub(',','', get_table.iloc[row,col]))                   
                            install_num.loc[str(get_pcode), str(get_year)] += float(re.sub(',','', get_table.iloc[row,col-1]))                   
    
    # save down the compiled install data
    install_data.to_csv(wrangled_data_path + 'install_data.csv')
    install_num.to_csv(wrangled_data_path + 'install_num.csv')

    
    # this section retrieves all the data that is used for anaylsis and combines 
    # into a single dataframe
    
    # retrieve viable building data
    buildings = pd.read_csv(wrangled_data_path + 'total_viable_buildings.csv', index_col=0, dtype=str)
    buildings = buildings.astype(int)
    # retrieve total building data
    total_buildings = pd.read_csv(wrangled_data_path + 'total_buildings.csv', index_col=0, dtype=str)
    total_buildings = total_buildings.astype(int)
    # retrieve installattion data
    install_num = pd.read_csv(wrangled_data_path + 'install_num.csv', index_col=0, dtype=str)
    install_num = install_num.astype(float)
    # retrieve income data
    total_income = pd.read_csv(wrangled_data_path + 'total_income.csv', index_col=0, dtype=str)
    total_income = total_income.astype(float)
    
    # combine data sets
    all_data = pd.melt(buildings.reset_index(), id_vars='region_id')
    all_data.columns=['postcode', 'year', 'viable_buildings']
    building_data = pd.melt(total_buildings.reset_index(), id_vars='region_id')
    building_data.columns=['postcode', 'year', 'total_buildings']
    all_data = all_data.merge(building_data, how='inner')
    income_data = pd.melt(total_income.reset_index(), id_vars='index')
    income_data.columns=['postcode', 'year', 'total_income']
    all_data = all_data.merge(income_data, how='inner')
    solar_data = pd.melt(install_num.reset_index(), id_vars='index')
    solar_data.columns=['postcode', 'year', 'num_installs']
    all_data = all_data.merge(solar_data, how='inner')
    all_data['average_income'] = all_data['total_income'] / all_data['total_buildings']
    all_data['install_percent'] = all_data['num_installs'] / all_data['viable_buildings']
    
    # save down the compiled data
    all_data.to_csv(wrangled_data_path + 'all_data_new.csv')
    

# 2. Analyse equitable uptake of rooftop solar
def equitable_uptake_solar():
    #  
    #  1) Split up postcode installation to deciles for each year
    # group postcodes by total installation / viable buildings per year
    a_df = pd.read_csv(wrangled_data_path + 'all_data_new.csv',encoding = 'ISO-8859-1')
    a_df = a_df.replace(np.nan, 0)
    a_df = a_df.replace(np.inf, 0)
    a_df = a_df[(a_df["num_installs"] != 0) & (a_df["average_income"] != 0)]
    a_df = a_df.loc[:,['year','postcode','num_installs', 'average_income']]
    a_df = a_df[(a_df["num_installs"] != 0) & (a_df["average_income"] != 0)]
    a_df['num_installs'] = np.floor(pd.to_numeric(a_df['num_installs'], errors='coerce')).astype('Int64')

    decile_df = pd.DataFrame()
    for year_n in range(2001, 2019):
        temp_df = a_df.loc[a_df['year']==year_n,:].copy()
        temp_df['decile'] = pd.qcut(temp_df['num_installs'], 10, duplicates = 'drop', labels = False)   

        decile_df = decile_df.append(temp_df)

    #  2) Analyse if the income composition has changed over time
    income_df = decile_df.filter(['year','decile','average_income'])
    income_df = income_df.groupby(['year','decile'])["average_income"].mean().to_frame(name = "average_income").reset_index()
    income_df= income_df.sort_values(by=['year', 'decile'])
    # Show in a table
    income_df.to_csv(wrangled_data_path + 'income_by_installation_decile.csv', index=False)

    #  3) Show visulisation (Heat map)
    decile_df = decile_df [decile_df ["year"] > 2008]
    heatmap = decile_df .pivot_table(index='decile', columns='year', values='average_income', aggfunc='mean')
    heatmap.sort_index(ascending=True)
    sns.heatmap(heatmap, cmap='viridis')
    plt.title("Annual Income by Installation Decile")
    plt.savefig(plots_path + 'Annual Income by Installation Decile on heatmap.png')
    
    # Flush memory space for plt
    plt.clf()
    plt.cla()
    plt.close()
    # 4) Look at standard deviation / variability of houehold income in the deciles
    sd_income= []
    decile = []
    for i in range(0,10):
        data = decile_df[(decile_df["decile"] == i)]
        decile.append(i)
        sd_income.append(data.std()['average_income'])

    sd = pd.DataFrame({'decile':decile,'sd_income':sd_income }) 
    sd.to_csv(wrangled_data_path + 'sd_income_per_decile.csv', index=False)
    

# 3. Overview of data change over time 
def data_change():

    # 3. Overview of data change over time  
    a_df2 = pd.read_csv(wrangled_data_path + 'all_data_new.csv',encoding = 'ISO-8859-1')
    a_df2 = a_df2.replace(np.nan, 0)
    a_df2 = a_df2.replace(np.inf, 0)
    a_df2['year'] = a_df2['year'].astype('str')
    a_df2 = a_df2.drop('postcode',1)
    a_df2 = a_df2.drop('Unnamed: 0',1)
    a_df2 = a_df2.drop('install_percent',1)

    a_df2.set_index('year', inplace=True)

    #  1) Calculate the mean per year and plot the relationship
    a_df2 = a_df2.groupby('year').mean()
    #  2) Normalise the data using min-max normalisation
    a_df2=(a_df2-a_df2.min())/(a_df2.max()-a_df2.min())
    a_df2.plot()
    plt.xlabel("Year")
    plt.title("Annual Income by Installation Decile\n")
    plt.savefig(plots_path + 'Annual Income by Installation Decile on lineplot.png')
    
    # Flush memory space for plt
    plt.clf()
    plt.cla()
    plt.close()

# 4. Analyse installation to building growth regression
def building_growth_vs_install():
    a_df3 = pd.read_csv(wrangled_data_path + 'all_data_new.csv',encoding = 'ISO-8859-1')
    a_df3 = a_df3.replace(np.nan, 0)
    a_df3 = a_df3.replace(np.inf, 0)
    a_df3 = a_df3.loc[:,['year','postcode','viable_buildings', 'total_buildings', 'num_installs']]
    a_df3 = a_df3[(a_df3["viable_buildings"] != 0)]
    a_df3.set_index('year', inplace=True)
    #  1) Calculate anual installation rate
    a_df3['i_rate']=a_df3['num_installs']/ a_df3['viable_buildings']
    #  2) Calculate anual viable buildings growth rate
    new = pd.read_csv(wrangled_data_path + 'total_viable_buildings.csv',encoding = 'ISO-8859-1')
    i_postcode = new['region_id'].to_list()
    new_postcode = []
    for i in range (0,17):
        new_postcode.extend(i_postcode)
    new.set_index('region_id', inplace=True)
    new = new.pct_change(axis='columns', periods=1)
    new = new.drop('2001',1)

    lst_i = []
    i_year = []
    for i in range(2,19):
        str_name = str(2000+i)
        for index, row in new.iterrows():

            lst_i.append(str(row[str_name]))
            i_year.append(int(str_name)-1)

    #  3) Merge viable buildings growth rate and installation rate dataframes
    new_i = pd.DataFrame({'postcode':new_postcode, 'year': i_year, 'vb_growth_rate': lst_i}) 
    reg_df = pd.merge(a_df3,new_i, on=["postcode","year"], how='inner')
    reg_df = reg_df.replace(np.nan, 0)
    reg_df = reg_df.replace(np.inf, 0)
    reg_df = reg_df.astype(np.float64)
    # - Get rid of all 0 values
    reg_df = reg_df[(reg_df['i_rate']!=0) & (reg_df['vb_growth_rate']!=0)]

    # - Get rid of outliers
    z_scores = stats.zscore(reg_df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    reg_df = reg_df[filtered_entries]
    reg_df = reg_df.loc[:,['year','vb_growth_rate','i_rate']]
    reg_df.set_index('year', inplace=True)

    reg_df.to_csv(wrangled_data_path + 'regression.csv', index=False)
    plt.scatter(reg_df.iloc[:,0],reg_df.iloc[:,1],color='green')
    plt.xlabel("annual growth rate of viable builings")
    plt.title("Installation to Building Growth Regression\n")
    plt.savefig(plots_path + 'Installation to building growth regression on scatterplot.png')
    
    # Flush memory space for plt
    plt.clf()
    plt.cla()
    plt.close()

def create_solar_pen_datasets():
    # retrieve viable building data
    buildings = pd.read_csv(wrangled_data_path + 'total_viable_buildings.csv', index_col=0, dtype=str)
    buildings = buildings.astype(int)
    # retrieve total building data
    total_buildings = pd.read_csv(wrangled_data_path + 'total_buildings.csv', index_col=0, dtype=str)
    total_buildings = total_buildings.astype(int)
    # retrieve installation data
    install_num = pd.read_csv(wrangled_data_path + 'install_num.csv', index_col=0, dtype=str)
    install_num = install_num.astype(float)
    # retrieve income data
    total_income = pd.read_csv(wrangled_data_path + 'total_income.csv', index_col=0, dtype=str)
    total_income = total_income.astype(float)
    #total_income.index = total_income.index.astype(str)
    
    
    
    # compile average income
    average_income = total_income / total_buildings
    
    # compile annual install_percent
    annual_install_per = install_num / buildings
    
    # combine data sets
    
    all_data = pd.melt(total_income.reset_index(), id_vars='index')
    all_data.columns=['postcode', 'year', 'total_income']
    
    
    viable = pd.melt(buildings.reset_index(), id_vars='region_id')
    viable.columns=['postcode', 'year', 'viable_buildings']
    
    all_data = all_data.merge(viable, how='left')
    
    building_data = pd.melt(total_buildings.reset_index(), id_vars='region_id')
    building_data.columns=['postcode', 'year', 'total_buildings']
    
    all_data = all_data.merge(building_data, how='left')
    
    #income_data = pd.melt(total_income.reset_index(), id_vars='index')
    #income_data.columns=['postcode', 'year', 'total_income']
    #all_data = all_data.merge(income_data, how='left')
    
    solar_data = pd.melt(install_num.reset_index(), id_vars='index')
    solar_data.columns=['postcode', 'year', 'num_installs']
    
    all_data = all_data.merge(solar_data, how='left')
    
    all_data['average_income'] = all_data['total_income'] / all_data['total_buildings']
    all_data['install_percent'] = all_data['num_installs'] / all_data['viable_buildings']
    
    all_data.to_csv(wrangled_data_path + 'all_data_new_sp.csv')
    
#creates a csv for cumulative solar installation
def solar_pen_cum_install():
    all_data = pd.read_csv(wrangled_data_path + 'all_data_new_sp.csv', index_col=0)
    all_data = all_data.fillna(0)
    post_dict = {}
    year_list = []
    file_list = []
    for index, row in all_data.iterrows():
        
        if not post_dict.get(row[0]):
            post_dict[row[0]] = {}
    
        post_dict[row[0]][row[1]] = row[5]
    
    year_list = post_dict[3000].keys()
    
    for postcode, dict1 in post_dict.items():
        for year in year_list:
            if year != 2001:
                post_dict[postcode][year] = post_dict[postcode][year] + post_dict[postcode][year-1]
                    
    for postcode, dict1 in post_dict.items():
        for year, installs in dict1.items():
            file_list.append({
                'postcode': postcode,
                'year': year,
                'cum_install': installs})
    
    headers = ['postcode', 'year', 'cum_install']
    with open(wrangled_data_path+ 'solar_cum_install.csv', 'w', newline= '') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames= headers)
            writer.writeheader()
            writer.writerows(file_list)
            
# creates a function to deal with diving by zero
def foo(x,y):
    if y ==0:
        return 0
    else:
        return ((x-y)/y)
    
# to merge cumulative data to the base solar pen data and caclucate a few other metrics    
def solar_pen_add_cum_install():
    all_data = pd.read_csv(wrangled_data_path + 'all_data_new_sp.csv').replace([np.inf, -np.inf], np.nan).fillna(0)
    cum_install = pd.read_csv(wrangled_data_path + 'solar_cum_install.csv').replace([np.inf, -np.inf], np.nan).fillna(0)
    datav1 = pd.merge(all_data, cum_install[['year', 'postcode', 'cum_install']], on = ['postcode', 'year'],  how='left')
    viable_avg = datav1.groupby(["postcode"]).viable_buildings.mean().reset_index(name='viable_avg')
    datav2 = pd.merge(datav1, viable_avg[['postcode', 'viable_avg']], on = ['postcode'],  how='left')

    datav2.loc[datav2['viable_buildings'] != 0, 'solar_pen'] = datav2['num_installs']/datav2['viable_buildings']
    datav2.loc[datav2['viable_buildings'] == 0, 'solar_pen'] = 0

    datav2.loc[datav2['viable_avg'] != 0, 'solar_pen_cum'] = datav2['cum_install']/datav2['viable_avg']
    datav2.loc[datav2['viable_avg'] <= 0, 'solar_pen_cum'] = 0

    solar_min = datav2.solar_pen_cum.min()
    solar_max = datav2[datav2['solar_pen_cum'] !=0].solar_pen_cum.max()
    datav2.loc[datav2['solar_pen_cum'] > 0, 'solar_pen_cum_n'] = (datav2['solar_pen_cum'] - solar_min)/(solar_max - solar_min)
    datav2.loc[datav2['solar_pen_cum'] <= 0, 'solar_pen_cum_n'] = 0
    
    #normalise avg_income
    income_min = datav2[datav2['average_income'] !=0].average_income.min()
    income_max = datav2[datav2['average_income'] !=0].average_income.max()
    datav2.loc[datav2['average_income'] > 0, 'average_income_n'] = (datav2['average_income'] - income_min)/(income_max - income_min)
    datav2.loc[datav2['average_income'] <= 0, 'average_income_n'] = 0
    
    datav2.to_csv(wrangled_data_path + 'all_data_new_sp_v2.csv', index = False) 
    
# to update base solar penetration table with yearly normalised data
def solar_pen_norm_yr():
    all_data = pd.read_csv(wrangled_data_path + 'all_data_new_sp_v2.csv').replace([np.inf, -np.inf], np.nan).fillna(0)
    solar_min_yr = all_data.groupby([ "year"]).solar_pen_cum.min().reset_index(name='solar_min_yr')
    solar_max_yr = all_data.groupby([ "year"]).solar_pen_cum.max().reset_index(name='solar_max_yr')
    datav1 = pd.merge(all_data, solar_min_yr, on = ['year'],  how='left')
    datav2 = pd.merge(datav1, solar_max_yr, on = ['year'],  how='left')
    datav2.loc[datav2['solar_pen_cum'] > 0, 'solar_pen_cum_n_yr'] = (datav2['solar_pen_cum'] - datav2['solar_min_yr'])/(datav2['solar_max_yr'] - datav2['solar_min_yr'])
    datav2.loc[datav2['solar_pen_cum'] <= 0, 'solar_pen_cum_n_yr'] = 0
    
    income_min_yr = all_data[all_data['average_income'] !=0].groupby([ "year"]).average_income.min().reset_index(name='income_min_yr')
    income_max_yr = all_data[all_data['average_income'] !=0].groupby([ "year"]).average_income.max().reset_index(name='income_max_yr')
    datav3 = pd.merge(datav2, income_min_yr, on = ['year'],  how='left')
    datav4 = pd.merge(datav3, income_max_yr, on = ['year'],  how='left')
    datav4.loc[datav4['average_income'] > 0, 'average_income_n_yr'] = (datav4['average_income'] - datav4['income_min_yr'])/(datav4['income_max_yr'] - datav4['income_min_yr'])
    datav4.loc[datav4['average_income'] <= 0, 'average_income_n_yr'] = 0
    
    datav4.to_csv(wrangled_data_path +  'all_data_new_sp_v3.csv', index = False) 
    



def income_bands_console():
    sns.set_theme()

    # Constants
    BRACKET_1 = 0
    BRACKET_2 = 50000
    BRACKET_3 = 100000
    BRACKET_4 = 150000
    
    INCOME_1 = "Lower (<50000)"
    INCOME_2 = "Middle-Lower (50000-100000)"
    INCOME_3 = "Middle-Upper (100000-150000)"
    INCOME_4 = "Upper (>150000)"
    
    LABEL_1 = "Installations per Viable Building\nacross Suburb (%)"
    LABEL_2 = "Installations across Suburb"
    
    TITLE_1 = "Solar Panel Installations by Income Group"
    TITLE_2 = "Cumulative Solar Panel Installations by Income Group"
    TITLE_3 = "Cumulative Solar Panel Installation % by Income Group"
    
    COLORS = {INCOME_1: "tab:red", INCOME_2: "tab:orange", INCOME_3: "tab:blue", INCOME_4: "tab:green"}
    YEARS = range(2001, 2019)
    
    # Data Wrangling
    data = pd.read_csv(wrangled_data_path + "all_data_new.csv").dropna()
    data = data[(data["viable_buildings"] != 0) & (data["total_buildings"] != 0) & (data["total_income"] != 0) & (data["viable_buildings"] >= data["num_installs"])]
    data["install_percent"] = data["install_percent"] * 100
    data["cum_num_installs"] = data.groupby("postcode")["num_installs"].cumsum()
    data["cum_install_percent"] = (data["cum_num_installs"] / data["viable_buildings"]) * 100
    data = data[data["cum_install_percent"] <= 100]
    data["income_group"] = np.select([data["average_income"].between(BRACKET_1, BRACKET_2, inclusive = "left"),
                                      data["average_income"].between(BRACKET_2, BRACKET_3, inclusive = "left"),
                                      data["average_income"].between(BRACKET_3, BRACKET_4, inclusive = "left")],
                                     [INCOME_1, INCOME_2, INCOME_3], default = INCOME_4)
    
    # Boxplot
    def boxplot(y_axis, title, percentage, filepath):
        for year in YEARS:
            year_data = data[data["year"] == year]
            sns.boxplot(data = year_data, whis = (0, 100), x = "income_group", y = y_axis, order = COLORS.keys(), palette = COLORS)
            plt.title(title.format(year), fontweight = "bold")
            plt.xlabel("Income Group", fontweight = "bold")
            if percentage:
                label = LABEL_1
            else:
                label = LABEL_2
            plt.ylabel(label, fontweight = "bold")
            plt.xticks(range(4), ["Lower\n(<50000)", "Middle-Lower\n(50000-100000)", "Middle-Upper\n(100000-150000)", "Upper\n(>150000)"], fontsize = 10)
            plt.savefig(plots_path + filepath + "boxplot-" + str(year) + ".png", bbox_inches = "tight")
            plt.clf()
    
    create_folder(plots_path + 'boxplots/absolute/')    
    create_folder(plots_path + 'boxplots/cumulative/')    
    create_folder(plots_path + 'boxplots/percentage/')    
    boxplot("num_installs", TITLE_1 + " ({})", False, "boxplots/absolute/")
    boxplot("cum_num_installs", TITLE_2 + " (2001-{})", False, "boxplots/cumulative/")
    boxplot("cum_install_percent", TITLE_3 + " (2001-{})", True, "boxplots/percentage/")
    
    # Catplot
    def catplot(y_axis, title, percentage, filepath):
        sns.catplot(data = data, x = "year", y = y_axis, hue = "income_group", palette = sns.color_palette(COLORS.values()), legend = False)
        plt.title(title, fontweight = "bold")
        plt.legend()
        plt.xlabel("Year", fontweight = "bold")
        if percentage:
            label = LABEL_1
        else:
            label = LABEL_2
        plt.ylabel(label, fontweight = "bold")
        plt.xticks(rotation = 90)
        plt.savefig(plots_path + filepath, bbox_inches = "tight")
        plt.clf()
    
    create_folder(plots_path + 'catplots/')
    catplot("num_installs", TITLE_1 + " (2001-2018)", False, "catplots/absolute.png")
    catplot("cum_num_installs", TITLE_2 + " (2001-2018)", False, "catplots/cumulative.png")
    catplot("cum_install_percent", TITLE_3 + " (2001-2018)", True, "catplots/percentage.png")
    
    # Dotplot
    def dotplot(y_axis, title, percentage, filepath):
        for year in YEARS:
            year_data = data[data["year"] == year]
            plt.scatter(year_data["average_income"] / 100000, year_data[y_axis])
            plt.title(title.format(year), fontweight = "bold")
            plt.xlabel("Average Income per Household (x100,000)", fontweight = "bold")
            if percentage:
                label = LABEL_1
            else:
                label = LABEL_2
            plt.ylabel(label, fontweight = "bold")
            plt.savefig(plots_path + filepath + "dotplot-" + str(year) + ".png", bbox_inches = "tight")
            plt.clf()
    
    create_folder(plots_path + 'dotplots/absolute/')    
    create_folder(plots_path + 'dotplots/cumulative/')    
    create_folder(plots_path + 'dotplots/percentage/')    
    dotplot("num_installs", "Solar Panel Installations and Income across Suburb ({})", False, "dotplots/absolute/")
    dotplot("cum_num_installs", "Cumulative Solar Panel Installations and Income across Suburb (2001-{})", False, "dotplots/cumulative/")
    dotplot("cum_install_percent", "Cumulative Solar Panel Installation % and Income across Suburb (2001-{})", True, "dotplots/percentage/")
    
    # Group Data by Income Group
    absolute_data = data.groupby(["year", "income_group"])["num_installs"].mean().unstack(fill_value = 0).stack().to_frame(name = "data")
    cumulative_data = data.groupby(["year", "income_group"])["cum_num_installs"].mean().unstack(fill_value = 0).stack().to_frame(name = "data")
    percentage_data = data.groupby(["year", "income_group"])["cum_install_percent"].mean().unstack(fill_value = 0).stack().to_frame(name = "data")
    
    absolute_data.to_csv(wrangled_data_path + "installations_by_income_group.csv")
    cumulative_data.to_csv(wrangled_data_path + "cumulative_installations_by_income_group.csv")
    percentage_data.to_csv(wrangled_data_path + "cumulative_installation%_by_income_group.csv")
    
    absolute_data = pd.read_csv(wrangled_data_path +  "installations_by_income_group.csv")
    cumulative_data = pd.read_csv(wrangled_data_path +  "cumulative_installations_by_income_group.csv")
    percentage_data = pd.read_csv(wrangled_data_path +  "cumulative_installation%_by_income_group.csv")
    
    # Histogram
    def histogram(data, title, percentage, filepath):
        plt.bar(YEARS, data[data["income_group"] == INCOME_1]["data"], color = COLORS[INCOME_1], label = INCOME_1)
        plt.bar(YEARS, data[data["income_group"] == INCOME_2]["data"], color = COLORS[INCOME_2], label = INCOME_2)
        plt.bar(YEARS, data[data["income_group"] == INCOME_3]["data"], color = COLORS[INCOME_3], label = INCOME_3)
        plt.bar(YEARS, data[data["income_group"] == INCOME_4]["data"], color = COLORS[INCOME_4], label = INCOME_4)
        
        #histogram_helper(data, INCOME_1, COLORS[INCOME_1])
        #histogram_helper(data, INCOME_2, COLORS[INCOME_2])
        #histogram_helper(data, INCOME_3, COLORS[INCOME_3])
        #histogram_helper(data, INCOME_4, COLORS[INCOME_4])
        plt.title(title, fontweight = "bold")
        plt.legend()
        plt.xlabel("Year", fontweight = "bold")
        if percentage:
            label = "Average Installations per Viable Building (%)"
        else:
            label = "Average Installations"
        plt.ylabel(label, fontweight = "bold")
        plt.xticks(YEARS, rotation = 90)
        plt.savefig(plots_path + filepath, bbox_inches = "tight")
        plt.clf()
    
    def histogram_helper(data, income_group, bar_color):
        plt.bar(YEARS, data[data["income_group"] == income_group]["data"], color = bar_color, label = income_group)
    
    create_folder(plots_path + 'histograms/')    
    histogram(absolute_data, TITLE_1 + " (2001-2018)", False, "histograms/absolute.png")
    histogram(cumulative_data, TITLE_2 + " (2001-2018)", False, "histograms/cumulative.png")
    histogram(percentage_data, TITLE_3 + " (2001-2018)", True, "histograms/percentage.png")
    
    # Pie Chart
    def pie(data, title, filepath):
        fig, axes = plt.subplots(3, 6)
        for year in YEARS:
            year_data = data[data["year"] == year]
            ax = axes[(year - 2001) // 6, (year - 2001) % 6]
            ax.pie(year_data["data"], normalize = True, colors = COLORS.values())
            ax.set_title(year, fontweight = "bold")
        fig.suptitle(title, fontweight = "bold")
        plt.legend(COLORS.keys(), bbox_to_anchor = (0, 0), loc = "upper center")
        plt.savefig(plots_path + filepath, bbox_inches = "tight")
        plt.clf()
    
    create_folder(plots_path + 'pie-charts/')
    pie(absolute_data, TITLE_1 + " (2001-2018)", "pie-charts/absolute.png")
    pie(cumulative_data, TITLE_2 + " (2001-2018)", "pie-charts/cumulative.png")
        

def solar_pen_plot_graph(geo_data , col_name, year, is_save, save_dir, titlez):
    #fig, ax = plt.subplots(1,1, figsize=(25,25))
    fig, ax = plt.subplots(1,1, figsize=(15,9))
    divider = make_axes_locatable(ax)
    tmp = geo_data.copy()
    cax = divider.append_axes("right", size="1%", pad=-1) 
    tmp.plot(column= col_name, ax=ax, cax=cax,  legend=True )
    tmp.geometry.boundary.plot(color='#BABABA', ax=ax, linewidth=0.3) 
    title = '{} - {}'.format(titlez, year)
    ax.set_title(title, fontdict={'fontsize': '20', 'fontweight': '3'})
    ax.axis('off')
    plt.tight_layout()
    if(is_save):
        plt.savefig(plots_path + save_dir + '{}.png'.format(title))

def solar_pen_heatmaps_viz():
    shape1 = gpd.read_file(raw_geo_path +  '/aus_poas.shp')
    all_data = gpd.read_file(wrangled_data_path + 'all_data_new2_v3.csv')
    
    shape1.rename(columns={'code': 'postcode'}, inplace=True)
    
    all_data['postcode'] = all_data['postcode'].astype('float').astype('Int64')
    all_data['total_buildings'] = all_data['total_buildings'].astype('float').astype('Int64')    
    all_data['solar_pen'] = all_data['solar_pen'].astype('float')
    all_data['solar_pen_cum_n'] = all_data['solar_pen_cum_n'].astype('float')
    all_data['solar_pen_cum_n_yr'] = all_data['solar_pen_cum_n_yr'].astype('float')
    all_data['average_income_n_yr'] = all_data['average_income_n_yr'].astype('float')
    all_data['average_income'] = all_data['average_income'].astype('float')
    all_data['average_income_n'] = all_data['average_income_n'].astype('float')
    
    all_data['average_income_n_yr_edit'] = all_data['average_income_n_yr']
    all_data['average_income_n_yr_edit'][(all_data['year'] == '2018') & (all_data['average_income_n_yr'] > 0.2)] = 0.2
    
    vic_shape = pd.merge(all_data[['postcode', 'year', 'solar_pen', 'solar_pen_cum_n', 'solar_pen_cum_n_yr', 'average_income',
                                   'average_income_n', 'average_income_n_yr',
                                   'average_income_n_yr_edit']], shape1[['geometry', 'postcode']], on = 'postcode',  how='left')
    
    create_folder(plots_path + '/heatmaps/average_income/')
    for i in range(18):
        year = 2001+i
        vic_shape1 = GeoDataFrame(vic_shape[vic_shape['year'] == '{}'.format(year)])
        solar_pen_plot_graph(vic_shape1, 'average_income_n_yr_edit', year, True, '/heatmaps/average_income/', 'Average Income per Household')
        
    create_folder(plots_path + '/heatmaps/solar_cum_pen/')
    for i in range(18):
        year = 2001+i
        vic_shape1 = GeoDataFrame(vic_shape[vic_shape['year'] == '{}'.format(year)])
        solar_pen_plot_graph(vic_shape1, 'solar_pen_cum_n_yr', year, True, '/heatmaps/solar_cum_pen/', 'Cumulative Solar Penetration')



def income_build_console():
    # retrieve viable building data
    buildings = pd.read_csv(wrangled_data_path + 'total_viable_buildings.csv', index_col=0, dtype=str)
    buildings = buildings.astype(int)
    # retrieve total building data
    total_buildings = pd.read_csv(wrangled_data_path + 'total_buildings.csv', index_col=0, dtype=str)
    total_buildings = total_buildings.astype(int)
    # retrieve installattion data
    install_num = pd.read_csv(wrangled_data_path + 'install_num.csv', index_col=0, dtype=str)
    install_num = install_num.astype(float)
    # cumulative installations by postcode
    total_installs = install_num.cumsum(axis=1)
    # retrieve income data
    total_income = pd.read_csv(wrangled_data_path + 'total_income.csv', index_col=0, dtype=str)
    total_income = total_income.astype(float)
    
    # compile average income
    average_income = total_income / total_buildings
    
    # compile annual install_percent
    annual_install_per = install_num / buildings
    
    # compile the income+building quantile data for a given year
    def get_year_data(year, num):
        # pull back data for this year and remove postcodes with missing data
        year_data = pd.concat([average_income[year], buildings[year],install_num[year],total_installs[year]], axis=1) 
        year_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        year_data = year_data.dropna()
        year_data = year_data.astype(int)
        # sort by average income and calculat install_percent, which is the percent of
        # viable households that installed solar in each postcode
        year_data.columns=['avg_income', 'viable_buildings','install_num', 'total_installs']
        year_data = year_data.sort_values(by='avg_income', ascending=True)        
        year_data['install_percent'] = year_data['install_num']/year_data['viable_buildings']
        # create deciles by quantiles an (approximately) equal share of buildings to each decile
        year_data['cum_total_buildings'] = year_data['viable_buildings'].cumsum()
        year_data['income_decile']=pd.cut(year_data['cum_total_buildings'],num, labels=False)
        year_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        year_data = year_data.dropna()
        return(year_data)
    
    # compile annual installation breakdown by income quantile
    def install_quantiles(average_income, buildings, install_num, num, label):    
        install_composition = pd.DataFrame(np.zeros([len(install_num.columns),num]))
        install_composition.columns = [label + str(num) for num in list(range(1,num+1))]
        install_composition.index = install_num.columns  
        # cumulative installations by decile
        cum_installs = install_composition.copy()        
        # track total number of viable buildings in each decile 
        total_buildings = install_composition.copy()    
        for year in install_composition.index:
            year_data = get_year_data(year, num)
            # compile share of installations
            # rescale factor used due to not exactly 10% of all viable buildings in each decile
            # install_percent is scaled to adjust for this
            rescale = (1/num)/(year_data.groupby(by='income_decile')['viable_buildings'].sum()/sum(year_data['viable_buildings']))
            install_share = year_data.groupby(by='income_decile')['install_num'].sum()/sum(year_data['install_num'])*rescale
            install_composition.loc[year,:] = install_share.to_frame().T.values
            # tally total viable buildings
            num_buildings = year_data.groupby(by='income_decile')['viable_buildings'].sum()
            total_buildings.loc[year,:] = num_buildings.to_frame().T.values
            # tally cumulative installs, adjusted by rescale factor
            num_installs = year_data.groupby(by='income_decile')['install_num'].sum()*rescale
            if year == install_composition.index[0]:
                cum_installs.loc[year,:] = num_installs.to_frame().T.values 
            else:
                cum_installs.loc[year,:] = cum_installs.loc[str(int(year)-1),:]+list(num_installs)       
        return (install_composition, cum_installs, total_buildings)
    
    # plots a stacked area chart showing share of installation by income quantile, by year
    def plot_stacked(install_composition, title_label, save_file):
        # get dimensions
        num_quantiles = len(install_composition.columns)
        first_date = install_composition.index[0]
        last_date = install_composition.index[len(install_composition.index)-1]
        label=re.sub('[0-9]*','',install_composition.columns[0])
        # set colour palette
        col = sns.color_palette("hls", num_quantiles)
        fig, ax = plt.subplots()
        ax.stackplot(install_composition.index,install_composition.to_numpy().T,
                     labels=install_composition.columns, colors=col)
        # horizontal lines at quantiles
        ax.hlines(np.arange(1/num_quantiles,1,1/num_quantiles),first_date,last_date, linestyles='dashed', colors='white')
        y_ticks = np.arange(0,1.1,1/num_quantiles)    
        ax.set_yticks(y_ticks)    
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig.set_size_inches(15,9)
        # move legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(title_label + 'Installation Composition by Income ' + label)
        ax.set_ylabel('Installation Percent of Total')
        plt.savefig(plots_path + save_file)
    
    # an alternative version that looks at how much each quantile was over or under their
    # fair share (=1/number of quantiles)
    def plot_diff(install_composition, title_label, save_file):
        # get dimensions
        num_quantiles = len(install_composition.columns)
        first_date = install_composition.index[0]
        last_date = install_composition.index[len(install_composition.index)-1]
        re.sub('[0-9]*','',install_composition.columns[0])
        label=re.sub('[0-9]*','',install_composition.columns[0])
        # set colour palette
        col = plt.cm.jet(np.linspace(0,1,num_quantiles))
        # calculate annual installation percent difference to "fair share"
        install_diff = install_composition-(1/num_quantiles)    
        fig, ax = plt.subplots()    
        for n in range(0,num_quantiles):
            ax.plot(install_diff.index,install_diff.iloc[:,n], color=col[n], linewidth = 2, label=install_diff.columns[n])
        ax.hlines(0, first_date, last_date, color='black', linestyles='dashed')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend()
        ax.set_ylabel('Installation Percent Difference to ' + str(int((1/num_quantiles)*100)) + '% ' + title_label)
        ax.set_title('Installation Percent Difference to ' + str(int((1/num_quantiles)*100)) + '% by Income ' + label + ', ' + title_label)
        fig.set_size_inches(15,9)
        plt.savefig(plots_path + save_file)
        
    
    # compile annual installation breakdown by income quantile
    def solar_pen_install_quantiles(average_income, buildings, install_num, num, label):    
        install_composition = pd.DataFrame(np.zeros([len(install_num.columns),num]))
        install_composition.columns = [label + str(num) for num in list(range(1,num+1))]
        install_composition.index = install_num.columns    
        # cumulative installations by postcode
        total_installs = install_num.cumsum(axis=1)
        # cumulative installations by decile
        cum_installs = install_composition.copy()        
        # track total number of viable buildings in each decile 
        total_buildings = install_composition.copy()
        total_install_k = install_composition.copy()    
        for year in install_composition.index:
            # pull back data for this year and remove postcodes with missing data
            year_data = pd.concat([average_income[year], buildings[year],install_num[year],total_installs[year]], axis=1) 
            year_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            year_data = year_data.dropna()
            year_data = year_data.astype(int)
            # sort by average income and calculat install_percent, which is the percent of
            # viable households that installed solar in each postcode
            year_data.columns=['avg_income', 'viable_buildings','install_num', 'total_installs']
            year_data = year_data.sort_values(by='avg_income', ascending=True)        
            year_data['install_percent'] = year_data['install_num']/year_data['viable_buildings']
            # create deciles by quantiles an (approximately) equal share of buildings to each decile
            year_data['cum_total_buildings'] = year_data['viable_buildings'].cumsum()
            year_data['income_decile']=pd.cut(year_data['cum_total_buildings'],num, labels=False)
            year_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            year_data = year_data.dropna()
            # compile share of installations
            # rescale factor used due to not exactly 10% of all viable buildings in each decile
            # install_percent is scaled to adjust for this
            rescale = (1/num)/(year_data.groupby(by='income_decile')['viable_buildings'].sum()/sum(year_data['viable_buildings']))
            install_share = year_data.groupby(by='income_decile')['install_num'].sum()/sum(year_data['install_num'])*rescale
            install_composition.loc[year,:] = install_share.to_frame().T.values
            # tally total viable buildings
            num_buildings = year_data.groupby(by='income_decile')['viable_buildings'].sum()
            total_buildings.loc[year,:] = num_buildings.to_frame().T.values
            #tall total installs
            num_installs_k = year_data.groupby(by='income_decile')['install_num'].sum()
            total_install_k.loc[year,:] = num_installs_k.to_frame().T.values
            # tally cumulative installs, adjusted by rescale factor
            num_installs = year_data.groupby(by='income_decile')['install_num'].sum()*rescale
            if year == install_composition.index[0]:
                cum_installs.loc[year,:] = num_installs.to_frame().T.values 
            else:
                cum_installs.loc[year,:] = cum_installs.loc[str(int(year)-1),:]+list(num_installs)       
        return (install_composition, cum_installs, total_buildings, total_install_k)
    
    # plots a stacked area chart showing share of installation by income quantile, by year
    def solar_pen_plot_stacked(install_composition, title_label, save_file):
        # get dimensions
        num_quantiles = len(install_composition.columns)
        first_date = install_composition.index[0]
        last_date = install_composition.index[len(install_composition.index)-1]
        label=re.sub('[0-9]*','',install_composition.columns[0])
        # set colour palette
        col = sns.color_palette("hls", num_quantiles)
        fig, ax = plt.subplots()
        ax.stackplot(install_composition.index,install_composition.to_numpy().T,
                     labels=install_composition.columns, colors=col)
        y_ticks = np.arange(0,1.1,1/num_quantiles)    
    
        fig.set_size_inches(15,9)
    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    
        ax.set_title(title_label + 'Solar Penetration by Income ' + label)
        ax.set_ylabel(title_label + 'Solar Penetration - Installs')
        plt.savefig(plots_path + save_file)
    
    
    # plot num of viable buildings, and average income per income decile for 2018
    year_data = get_year_data('2018', 10)
    num_postcodes_2018 = list(year_data.groupby(by='income_decile')['avg_income'].count().values)
    avg_income_2018 = list(year_data.groupby(by='income_decile')['avg_income'].mean().round(0).values)
    install_num_2018 = list(year_data.groupby(by='income_decile')['install_num'].sum().values)
    num_viable_2018 = list(year_data.groupby(by='income_decile')['viable_buildings'].sum().values)
    
    fig = go.Figure(data=[go.Table(header=dict(values=['income_decile', 'num_postcodes','average_income', 'install_num', 'viable_buildings']),
                     cells=dict(values=[list(np.arange(1,11)), num_postcodes_2018, avg_income_2018, install_num_2018, num_viable_2018]))])
    fig.write_image(plots_path + "Table2018.png")
    
    # run decile anlaysis
    install_deciles, cum_deciles, building_deciles = install_quantiles(average_income, buildings, install_num, 10, 'Decile')
    # turn cum installs into a percent share table
    cum_deciles = cum_deciles.div(cum_deciles.sum(axis=1), axis=0) 
    plot_stacked(install_deciles, 'Annual ', 'annual_decile.png')
    plot_stacked(cum_deciles, 'Cumulative ', 'cum_decile.png')
    plot_diff(install_deciles, 'Annual', 'annual_decile_diff.png')
    plot_diff(cum_deciles, 'Cumulative', 'cum_decile_diff.png')
    
    # run quintile analysis
    install_quintiles, cum_quintiles, building_quintiles = install_quantiles(average_income, buildings, install_num, 5, 'Quintile')
    # turn cum installs into a percent share table
    cum_quintiles = cum_quintiles.div(cum_quintiles.sum(axis=1), axis=0) 
    plot_stacked(install_quintiles, 'Annual ', 'annual_quintile.png')
    plot_stacked(cum_quintiles, 'Cumulative ', 'cum_quintile.png')
    plot_diff(install_quintiles, 'Annual', 'annual_quintile_diff.png')
    plot_diff(cum_quintiles, 'Cumulative', 'cum_quintile_diff.png')
    
    
    # to get solar penetration plots
    install_deciles, cum_deciles, building_deciles, total_install_k = solar_pen_install_quantiles(average_income, buildings, install_num, 10, 'Decile')
    solar_pen_cum = cum_deciles/building_deciles
    solar_pen_cum_n =(solar_pen_cum - solar_pen_cum.min())/(solar_pen_cum.max()-solar_pen_cum.min())
    solar_pen = total_install_k/building_deciles
    
    solar_pen_plot_stacked(solar_pen, 'Annual ', 'Annual Solar Penetration by Income Decile')
    solar_pen_plot_stacked(solar_pen_cum_n/10, 'Cumulative ', 'Cumulative Solar Penetration by Income Decile')






#-----------------------------------------------------------------------Main Console---------------------------------------------
create_buildings_data()
create_wrangled_data()
equitable_uptake_solar()
data_change()
building_growth_vs_install()
create_solar_pen_datasets()
solar_pen_cum_install()
solar_pen_add_cum_install()

solar_pen_norm_yr()
income_bands_console()
income_build_console()

if(IS_GEOPANDAS):
    import geopandas as gpd
    from geopandas import GeoDataFrame
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.pyplot import figure
    import matplotlib
    solar_pen_heatmaps_viz()
    
