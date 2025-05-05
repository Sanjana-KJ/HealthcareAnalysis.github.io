import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings as wr

wr.filterwarnings('ignore')


df = pd.read_csv("hospital_patient.csv")
print(df.head())

print(df.columns.tolist())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())


# Barplot for Patients Admission By Hour
df['admit_time'] = pd.to_datetime(df['admit_time'])
df['admission_hour'] = df['admit_time'].dt.hour
hourly_admissions = df.groupby('admission_hour').size()

plt.figure(figsize=(10, 6))
hourly_admissions.plot(kind='bar', color='skyblue',edgecolor='black')
plt.title('Patient Admissions by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
ax=plt.gca()
ax.set_facecolor('lightyellow')
plt.show()



#  Histplot for patients distribution by Wait Times
print(df['wait_time_mins'].describe())
plt.figure(figsize=(10,6))
sns.histplot(df['wait_time_mins'],bins=30,color="purple",edgecolor='black',alpha=0.7,kde =True)
plt.title('Distribution of wait time')
plt.xlabel('Wait Time (minutes)')
plt.ylabel('Patient Count')
plt.grid(True)
ax=plt.gca()
ax.set_facecolor('lightyellow')
plt.show()



# Barplot for  Patient Overcrowding by Weather
df['overcrowded'] = (df['wait_time_mins'] > 30).astype(int)
weather_overcrowding = df.groupby('weather')['overcrowded'].mean().reset_index()
print(weather_overcrowding)

plt.figure(figsize=(8, 5))
sns.barplot(data=weather_overcrowding, x='weather', y='overcrowded',hue='weather', palette='Set3')
plt.title('Average Overcrowding Rate by Season')
plt.ylabel('Overcrowding Rate')
plt.xlabel('Season')
plt.grid(True)
ax=plt.gca()
ax.set_facecolor('lightgreen')
plt.show()



#  violinplot for wait time distribution by Triage level
triage_waittime = df.groupby('triage_level')['wait_time_mins'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='triage_level', y='wait_time_mins', palette='Dark2',hue=None,legend=False)
plt.title('Wait Time Distribution by Triage Level')
plt.xlabel('Triage Level')
plt.ylabel('Wait Time (minutes)')
plt.grid(True)
plt.tight_layout()
plt.show()


# Baxplot for length of stay by Day of week
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='day_of_week', y='length_of_stay_mins', palette='Dark2')
plt.title('Length of Stay by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Length of Stay (minutes)')
plt.tight_layout()
plt.show()



# correlation analysis between wait_time_mins', 'length_of_stay_mins', 'overcrowded', 'hour_of_day', 'flu_outbreak_level
correlation_matrix = df[['wait_time_mins', 'length_of_stay_mins', 'overcrowded', 'hour_of_day', 'flu_outbreak_level']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',center=0,square=True, fmt='.2f', linewidths=0.5,cbar_kws={"shrink":0.8},annot_kws={"size":10})
plt.title("Correlation Heatmap of Key Features")
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()


#Predict ER Demand using Linear LinearRegression

df['admit_time'] = pd.to_datetime(df['admit_time'])
df['day_of_week_numeric'] = df['admit_time'].dt.dayofweek

features = ['hour_of_day', 'day_of_week_numeric']  
X = df[features]  
y = df['wait_time_mins']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R² Score: {r2}')
print(f'Mean Squared Error (MSE): {mse}')

# Predicting wait times using linear regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Wait Times")
plt.xlabel("Actual Wait Times (minutes)")
plt.ylabel("Predicted Wait Times (minutes)")
plt.grid(True)
ax = plt.gca()  
ax.set_facecolor('lightgreen') 
plt.gcf().set_facecolor('lightyellow') 
plt.show()


#Preddicting Time series using Prophet
df_forecast = df.groupby('date').size().reset_index(name='patient_count')
df_forecast.rename(columns={'date': 'ds', 'patient_count': 'y'}, inplace=True)
model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=30)  
forecast = model.predict(future)
model.plot(forecast)
plt.title("ER Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Patient Count")
plt.grid(True)
plt.show()


#ARIMA model for time series forecasting
df['date'] = pd.to_datetime(df['date'])
df_arima = df.groupby('date')['wait_time_mins'].size()
df_arima.index = pd.to_datetime(df_arima.index)  
df_arima = df_arima.asfreq('D')  
df_arima = df_arima.fillna(0)
arima_model = ARIMA(df_arima, order=(5, 1, 0))  
arima_model_fit = arima_model.fit()
forecast = arima_model_fit.forecast(steps=30)

plt.figure(figsize=(10, 6))
plt.plot(df_arima, label="Historical Data")
plt.plot(forecast, label="Forecast", color='red')
plt.title("Forecast of Daily Average ER Wait Times")
plt.xlabel("Date")
plt.ylabel("Average Wait Time (mins)")
plt.legend()
plt.tight_layout()
plt.show()



#Predicting High Risk Patients 
df['high_risk'] = ((df['wait_time_mins'] > 30) | (df['triage_level'] == 'Critical')).astype(int)
features = ['hour_of_day', 'day_of_week_numeric']
X = df[features]
y = df['high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Classification Report:")
print(classification_report(y_test, y_pred))