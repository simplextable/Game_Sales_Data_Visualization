import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML
pd.set_option("display.max_rows",22200)
pd.set_option("display.max_column",22200)


df = pd.read_csv('train.csv')


print(df.head(5))
print(df.shape)        
print(df.isnull().sum()) 
print(df.info())   


#Shows Missing Data as Percent
sns.set_style("whitegrid")
missing = df.isnull().sum()
missing = (missing[missing > 0]) / 16719
missing.sort_values(inplace=True)
missing.plot.bar()


# Annotated heatmap of all correlation coefficients between variables
plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()


#####################################################################################################################################
#I visualized how many rows there are according to platform type


total_data = df.dropna(subset=['Critic_Score', "Developer" ,"Critic_Count" ,"User_Score","User_Count","Rating"])
groupby_platform = total_data.groupby(['Platform']).size().reset_index()
all_platform=df.groupby(['Platform']).size().reset_index()

groupby_platform.plot.bar(x="Platform", figsize= (15, 7))
plt.show()


# This figure is same with previous one just metric is percent
d = {'Platform': [], 'Yüzde': []}  
final = pd.DataFrame(data = d)


for i in range(len(all_platform)):
    for j in range(len(groupby_platform)):
        if all_platform["Platform"][i] == groupby_platform["Platform"][j]:
           b = groupby_platform[0][j] / all_platform[0][i] * 100
           final.at[j , "Platform"] = groupby_platform["Platform"][j]
           final.at[j , "Percent"] = b
         
            
final.plot.barh(x = "Platform" , figsize=(15,7) , color = "red", title = "% ")
plt.show()


#####################################################################################################################################
# Percent of Published Game Types in the Market

genre_df =  df[["Genre"]]
genre_group =  genre_df.groupby(['Genre']).size().reset_index()

genre_group.plot(kind='pie', y = 0 , autopct='%1.1f%%',  title = "Published Game Types",
startangle=90, shadow=False, labels=genre_group['Genre'], legend = False, fontsize=14, figsize=(10, 10))

plt.show()


#####################################################################################################################################
# Game Type Percents of Published Games

gs_gnr_df =  df[["Genre","Global_Sales"]]
gb_group_sum =  gs_gnr_df.groupby(['Genre']).sum().reset_index()
gb_group_mean =  gs_gnr_df.groupby(['Genre']).mean().reset_index()


gb_group_sum.plot(kind='pie', y = "Global_Sales" , autopct='%1.1f%%',  title = "Game Type Percents of Published Games",startangle=90,
 shadow=False, labels=gb_group_sum['Genre'], legend = False, fontsize=14, figsize=(10, 10))
plt.show()


# Game Sales Percents According to Game Type
gb_group_mean.plot(kind='pie', y = "Global_Sales" , autopct='%1.1f%%',  title = "Game Sales Percents According to Game Type",startangle=90,
 shadow=False, labels=gb_group_mean['Genre'], legend = False, fontsize=14, figsize=(10, 10))
plt.show()


#####################################################################################################################################
#Disburution of Global Sales Which Game Type is Platform 

df_globalsales_platform =  gs_gnr_df[gs_gnr_df.Genre == "Platform"]    
sns.distplot(df_globalsales_platform['Global_Sales']);   
fig = plt.figure()
res = stats.probplot(df_globalsales_platform['Global_Sales'], plot=plt)
plt.show()

# This Graphic is Same With Previous One but Only Lesser than Seven Million Sold Games 

df_genre_global_7 = gs_gnr_df[gs_gnr_df["Global_Sales"]<7]
sns.distplot(df_genre_global_7['Global_Sales']);   
fig = plt.figure()
res = stats.probplot(df_genre_global_7['Global_Sales'], plot=plt)
plt.show()

# This Graphic is Same With Previous One but Only Lesser than One Million Sold Games 

df_genre_global_1 = gs_gnr_df[gs_gnr_df["Global_Sales"]<1]
sns.distplot(df_genre_global_1['Global_Sales']);   
fig = plt.figure()
res = stats.probplot(df_genre_global_1['Global_Sales'], plot=plt)
plt.show()


# This Graphic is Same With Previous One but Only Lesser than 0.2 Million Sold Games 

df_genre_global_0_2 = gs_gnr_df[gs_gnr_df["Global_Sales"]<0.2]
sns.distplot(df_genre_global_0_2['Global_Sales']);   
fig = plt.figure()
res = stats.probplot(df_genre_global_0_2['Global_Sales'], plot=plt)
plt.show()


#Distribution of Sold Games in All Game Types

sns.distplot(gs_gnr_df['Global_Sales']);   
fig = plt.figure()
res = stats.probplot(gs_gnr_df['Global_Sales'], plot=plt)
plt.show()


#####################################################################################################################################
#Groupped According to Global Sales in Some Levels

df_genre_global_grupby = gs_gnr_df.copy()
df_genre_global_grupby['grup'] = pd.Series(len(df_genre_global_grupby['Global_Sales']), index=df_genre_global_grupby.index)
df_genre_global_grupby['grup'] = 0 

df_genre_global_grupby.loc[ df_genre_global_grupby['Global_Sales'] <= 0.05, 'grup']= 0
df_genre_global_grupby.loc[(df_genre_global_grupby['Global_Sales'] > 0.05) & (df_genre_global_grupby['Global_Sales'] <= 0.1), 'grup'] = 1
df_genre_global_grupby.loc[(df_genre_global_grupby['Global_Sales'] > 0.1) & (df_genre_global_grupby['Global_Sales'] <= 0.2), 'grup']= 2
df_genre_global_grupby.loc[(df_genre_global_grupby['Global_Sales'] > 0.2) & (df_genre_global_grupby['Global_Sales'] <= 0.5), 'grup']= 3
df_genre_global_grupby.loc[(df_genre_global_grupby['Global_Sales'] > 0.5) & (df_genre_global_grupby['Global_Sales'] <= 1), 'grup']= 4
df_genre_global_grupby.loc[(df_genre_global_grupby['Global_Sales'] > 1) & (df_genre_global_grupby['Global_Sales'] <= 5), 'grup']= 5
df_genre_global_grupby.loc[ df_genre_global_grupby['Global_Sales'] > 5, 'grup']= 6

print(df_genre_global_grupby.groupby("grup").count())

sns.distplot(df_genre_global_grupby['grup']);   
fig = plt.figure()
res = stats.probplot(df_genre_global_grupby['grup'], plot=plt)
plt.show()

df_genre_global_grupby['grup'] = df_genre_global_grupby['grup'].astype('str')
df_genre_global_grupby.info()

#####################################################################################################################################
# Number of Sold Games According to Years 

yearly_sales =  df[["Name","Year_of_Release"]]
yearly_sales_df =yearly_sales.groupby(['Year_of_Release']).count()

yearly_global = df[["Name","Year_of_Release","Global_Sales"]]
yearly_global_df = yearly_global.groupby(["Year_of_Release"])[["Global_Sales"]].sum()

sales_yearly=pd.concat([yearly_global_df,yearly_sales_df ],axis=1).reset_index()

sales_yearly.plot.line(x='Year_of_Release', y='Global_Sales')
plt.show()

# Number of Published Games According to Years 
sales_yearly.plot.line(x='Year_of_Release', y='Name')
plt.show()


############################################################################################################################################3
# Console Preferences According To Continents


toplam_platform_NA_EU_JP=df.groupby(['Platform'])[["NA_Sales","EU_Sales","JP_Sales"]].sum().reset_index()
a = toplam_platform_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]] / toplam_platform_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]].sum()

print(toplam_platform_NA_EU_JP)
plat_sum = df[["Platform"]]
plat_sum = plat_sum.groupby(["Platform"]).size().reset_index()

deneme = pd.concat([a,plat_sum["Platform"]],axis=1)

plt.figure(figsize=(30,5))
line = sns.lineplot(data=deneme, x="Platform", y="NA_Sales", marker="o", label= "NA_Sales")
line = sns.lineplot(data=deneme, x="Platform", y="EU_Sales", marker="o", label= "EU_Sales")
line = sns.lineplot(data=deneme, x="Platform", y="JP_Sales", marker="o", label= "JP_Sales")

line.set(xticks=deneme.Platform.values)
line.set(ylabel="Percent")
plt.show()

#####################################################################################################################################
#Game Type Preferences According To Continents


toplam_genre_NA_EU_JP=df.groupby(['Genre'])[["NA_Sales","EU_Sales","JP_Sales"]].sum().reset_index()
toplam_genre = toplam_genre_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]] / toplam_genre_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]].sum()

print(toplam_genre_NA_EU_JP)
genre_sum = df[["Genre"]]
genre_sum = genre_sum.groupby(["Genre"]).size().reset_index()

genre_sales = pd.concat([toplam_genre,genre_sum["Genre"]],axis=1)

plt.figure(figsize=(30,5))
line = sns.lineplot(data=genre_sales, x="Genre", y="NA_Sales", marker="o", label= "NA_Sales")
line = sns.lineplot(data=genre_sales, x="Genre", y="EU_Sales", marker="o", label= "EU_Sales")
line = sns.lineplot(data=genre_sales, x="Genre", y="JP_Sales", marker="o", label= "JP_Sales")

line.set(xticks=genre_sales.Genre.values)
line.set(ylabel="Percent")
plt.show()igsize=(30,5))
line = sns.lineplot(data=genre_sales, x="Genre", y="NA_Sales", marker="o", label= "NA_Sales")
line = sns.lineplot(data=genre_sales, x="Genre", y="EU_Sales", marker="o", label= "EU_Sales")
line = sns.lineplot(data=genre_sales, x="Genre", y="JP_Sales", marker="o", label= "JP_Sales")

line.set(xticks=genre_sales.Genre.values)
line.set(ylabel="Percent")
plt.show()
