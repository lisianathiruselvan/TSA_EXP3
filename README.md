# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
Date: 

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/content/World Population.csv')

# Extract Year from "Date" column (taking first two characters and converting to full year)
data['Year'] = data['Date'].astype(str).str[:2].astype(int) + 2000  # Assuming years are 2000+

# Remove "%" from the "Percentage" column and convert it to a numeric value
data['Percentage'] = data['Percentage'].str.replace('%', '').astype(float)

# Group by Year and compute the mean percentage (assuming world total percentage remains ~100%)
resampled_data = data.groupby('Year', as_index=False)['Percentage'].mean()

# Extract values
years = resampled_data['Year'].tolist()
percentage = resampled_data['Percentage'].tolist()

# Preprocessing for Trend Calculation
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, percentage)]

# Polynomial Trend Estimation (Degree 2)
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, percentage)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(percentage), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]
# Print trend equations
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Add trends to dataset
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

# Set index to 'Year'
resampled_data.set_index('Year', inplace=True)

# Visualization
resampled_data['Percentage'].plot(kind='line', color='blue', marker='o', label='Actual Percentage')
resampled_data['Linear Trend'].plot(kind='line', color='black', linestyle='--', label='Linear Trend')
resampled_data['Polynomial Trend'].plot(kind='line', color='red', marker='o', label='Polynomial Trend')

plt.xlabel('Year')
plt.ylabel('Population Percentage (%)')
plt.legend()
plt.title('World Population Percentage Trend Estimation')
plt.grid()
plt.show()+

### OUTPUT:
![image](https://github.com/user-attachments/assets/39bc982d-3e72-4ac3-bb6d-871af0dd3fcd)

### RESULT:
        Thus we have successfully implemented the auto correlation function in python.
