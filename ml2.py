#Import required liabraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Read data from Mail_Customers.csv
df=pd.read_csv('Mall_Customers.csv')
#Print first few rowa of the data
df.head()
#Get some summary statistics of the data
df.describe()
#Check data types and missing values for each column
df.info()

#Filter for customers with spending score above 50
mask = df['Spending Score (1-100)']>50
df_score = df[mask]
#Print first few rows of the high spenders
df_score.head()
#Get some summary statistics of the high spendars
df_score.describe()

#Creating a figure of subplots with adjustable spacing
plt.figure(figsize = (15,6))
n=0
#Loop through the columns 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n +=1
    plt.subplot(2,3,n) #Set subplot location in the figure
    plt.subplots_adjust(hspace=0.2,wspace=0.2) #Adjust spacing between subplots
    #Create a histogram for each variable with 20 bins
    sns.histplot(df[x],bins = 20)
    plt.title('Distplot of ()'.format(x))
#Display the figure with all subplots
plt.show();

#Create a separate histogram specifically for high spenders' age distribution
df_score['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Spending Score(51 ~ 100): Age Distribution');

#Count plot for gender distribution among high spenders
plt.figure(figsize = (15,4))
sns.countplot(y='Gender',data=df_score)
plt.title('Spending Score (51-100): Gender Distribution')
plt.show();

#Count plot for gender distribution overall
plt.figure(figsize = (15,4))
sns.countplot(y='Gender',data=df_score)
plt.title('Spending Score (0-100): Gender Distribution')
plt.show();

#Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore' , category = UserWarning)
# Create a pairplot showing relationships between Age, Income, and Spending Score
sns.pairplot(df[['Age', 'Annual Income (k$)' , 'Spending Score (1-100)']],kind='reg')
plt.tight_layout()
plt.show();

#Scatter plot comparing Age vs Annual Income for both genders
plt.figure(1,figsize=(15,6))
for gender in ['Male','Female']:
    plt.scatter(x='Age',y='Annual Income (k$)',data=df[df['Gender']==gender],s=200,alpha=0.7,label=gender)
plt.xlabel('Age'),plt.ylabel('Annual Income (k$)')
plt.title('Age vs Annual Income wrt Gender')
plt.legend()
plt.show()

#Scatter plot comparing Annual Income vs Spending Score for both genders
plt.figure(1, figsize=(15, 6))
plt.figure(1,figsize=(15,6))
for gender in ['Male','Female']:
    plt.scatter(x='Annual Income (k$)',y='Spending Score (1-100)',data=df[df['Gender']==gender],s=200,alpha=0.7,label=gender)
plt.xlabel('Annual Income (k$)'),plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income vs Spending Salary wrt Gender')
plt.legend()
plt.show()

plt.figure(1,figsize = (15,6))
n = 0
for cols in ['Age','Annual Income (k$)', 'Spending Score (1-100)']:
    n +=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace = 0.3,wspace = 0.3)
    sns.violinplot(x=cols,y='Gender',data=df,palette='vlag')
    sns.swarmplot(x=cols,y='Gender',data=df)
    plt.ylabel('Gender' if n ==1 else '')
    plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show();

#Split
x=df.iloc[:,[3,4]]
print(f"x shape {x.shape}")
x.head()

n_clusters=range(2,13)
inertia_errors=[]
#Add a for loop to train model and calculate inertia,silhoutte score
silhoutte_scores = []
for k in n_clusters:
    model=KMeans(n_clusters = k,random_state=42,n_init =10)
    #Train model
    model.fit(x)
    #Calculate inertia
    inertia_errors.append(model.inertia_)
    #calculate silhoutte score
    silhoutte_scores.append(silhouette_score(x,model.labels_))
print("Inertia:",inertia_errors[:3])
print()
print("Silhoutte Scores:",silhoutte_scores[:3])

#Create a line plot of inertia_errors vs n_clusters
x_values = list(range(2,13))

plt.figure(figsize=(8,6))
sns.set(style="whitegrid") #sea seaborn style

#Create a line plot using Matplotlib
plt.plot(x_values,inertia_errors,marker='o',linestyle='-',color='b')

#Add labels and title
plt.title('K-Means Model: Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

#Turn on grid and show plot
plt.grid(True)
plt.show()

#Create a line plot of silhoutte scores vs n_clusters
x_values = list(range(2,13))

plt.figure(figsize=(8,6))
sns.set(style="whitegrid")

#Create a line plot using Matpotlib 
plt.plot(x_values,silhoutte_scores,marker='o',linestyle='-',color='b')

#Add labels and title
plt.title('K-Means Model: Silhoutte Scores vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhoutte Scores')

#Turn on grid and show plot
plt.grid(True)
plt.show()

final_model = KMeans(n_clusters=5,random_state=42,n_init=10)
final_model.fit(x)

labels = final_model.labels_
centroids = final_model.cluster_centers_
print(labels[:5])
print(centroids[:5])

#Plot "Annual Income" vs "Spending Score" with final_model labels 
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],hue=labels,palette='deep')
sns.scatterplot(
    x=centroids[:,0],
    y=centroids[:,1],
    color='black',
    marker='+',
    s=500)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score");

xgb = x.groupby(final_model.labels_).mean()
xgb

#Create side-by-side bar chart of 'xgb'
plt.figure(figsize=(8,6))

x=[0,1,2,3,4]
x_labels=labels
income_values=xgb['Annual Income (k$)']
spending_values=xgb['Spending Score (1-100)']

bar_width=0.35
index=range(len(x))

#Create grouped bar plot using Matplotlib
plt.bar(index,income_values,bar_width,label='Annual Income')
plt.bar([i+bar_width for i in index],spending_values,bar_width,label='Spending Score')

#Add labels and title
plt.xlabel('Clusters')
plt.ylabel('Value')
plt.title('Annual Income and Spending Score by Cluster')
plt.xticks([i+bar_width/2 for i in index],x)
plt.legend()

#Show plot
plt.show()
