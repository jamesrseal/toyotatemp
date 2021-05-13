import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# curve fitting function for scipy.optimize.curve_fit
def func(x, a, b):
    return a * np.exp(b * x)

# get the data from the csv file
temp_data = np.genfromtxt('Toyota Temp Sensor Analysis - Combined Data.csv', delimiter=',', skip_header=1)

# put the delimited data into [1,1] arrays to prep for plotting ignore the first two rows
# and last 4 rows of data because there are no data points at these temps and curve fitting will throw and error
stock_2_wire = temp_data[2:76, 2]
aftermarket_2_wire = temp_data[2:76,3]
aftermarket_3_wire = temp_data[2:76,4]

# reference column 0 to plot data in Celsius e.g. temp = temp_data[2:76, 0]
# plt.xticks(np.arange(20, 100, 10.0)) should also be adjusted
temp = temp_data[2:76, 1]

# curve fit the stock 2 wire temp sensor data
optimizedParameters, pcov = opt.curve_fit(func, temp, stock_2_wire, p0=[20000, 0], bounds=(-np.inf, np.inf))
residuals = stock_2_wire - func(temp, *optimizedParameters)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((stock_2_wire-np.mean(stock_2_wire))**2)
stdevs = np.sqrt(np.diag(pcov))
r_squared = 1 - (ss_res / ss_tot)
r_squared_text = 'R$^2$ = ' + str(round(r_squared,4))
fit_equation = 'y = ' + str(round(optimizedParameters[0],2)) + ' * exp(' + str(round(optimizedParameters[1],3)) + 'x)\n' + r_squared_text


# curve fit the aftermarket 2 wire temp sensor data
optimizedParameters1, pcov1 = opt.curve_fit(func, temp, aftermarket_2_wire, p0=[20000, 0], bounds=(-np.inf, np.inf))
residuals1 = aftermarket_2_wire - func(temp, *optimizedParameters1)
ss_res1 = np.sum(residuals1**2)
ss_tot1 = np.sum((aftermarket_2_wire-np.mean(aftermarket_2_wire))**2)
stdevs1 = np.sqrt(np.diag(pcov1))
r_squared1 = 1 - (ss_res1 / ss_tot1)
r_squared_text1 = 'R$^2$ = ' + str(round(r_squared,4))
fit_equation1 = 'y = ' + str(round(optimizedParameters1[0],2)) + ' * exp(' + str(round(optimizedParameters1[1],3)) + 'x)\n' + r_squared_text1

# curve fit the aftermarket 3 wire temp sensor data
optimizedParameters2, pcov2 = opt.curve_fit(func, temp, aftermarket_3_wire, p0=[20000, 0], bounds=(-np.inf, np.inf))
residuals2 = aftermarket_3_wire - func(temp, *optimizedParameters2)
ss_res2 = np.sum(residuals2**2)
ss_tot2 = np.sum((aftermarket_3_wire-np.mean(aftermarket_3_wire))**2)
stdevs2 = np.sqrt(np.diag(pcov2))
r_squared2 = 1 - (ss_res2 / ss_tot2)
r_squared_text2 = 'R$^2$ = ' + str(round(r_squared,4))
fit_equation2 = 'y = ' + str(round(optimizedParameters2[0],2)) + ' * exp(' + str(round(optimizedParameters2[1],3)) + 'x)\n' + r_squared_text2

# plot the stock 2 wire sensor data
fig1, ax1 = plt.subplots()
ax1.plot(temp, stock_2_wire, label='raw sensor data')
ax1.plot(temp, func(temp, *optimizedParameters), label=fit_equation, linestyle='--', linewidth=1, color='black')
ax1.legend()
plt.xlabel('Temperature ($^\circ$F)')
plt.ylabel('Resistance ($\Omega$)')
plt.title('Stock 2 Wire Resistance vs Temperature Profile')
plt.grid(True, which = 'both')
plt.xticks(np.arange(60, 225, 10.0))
plt.xticks(rotation = 45)
plt.yscale("log")
plt.yticks([1000, 10000, 100000])
plt.subplots_adjust(top = .95, bottom = 0.15)
plt.show()

# plot the aftermarket 2 wire sensor data
fig2, ax2 = plt.subplots()
ax2.plot(temp, aftermarket_2_wire, label='raw sensor data', color = 'g')
ax2.plot(temp, func(temp, *optimizedParameters1), label=fit_equation1, linestyle='--', linewidth=1, color='black')
ax2.legend()
plt.xlabel('Temperature ($^\circ$F)')
plt.ylabel('Resistance ($\Omega$)')
plt.title('Aftermarket 2 Wire Resistance vs Temperature Profile')
plt.grid(True, which = 'both')
plt.xticks(np.arange(60, 225, 10.0))
plt.xticks(rotation = 45)
plt.yscale("log")
plt.yticks([1000, 10000, 100000])
plt.subplots_adjust(top = .95, bottom = 0.15)
plt.show()

# plot the afermarket 3 wire sensor data
fig3, ax3 = plt.subplots()
ax3.plot(temp, aftermarket_3_wire, label='raw sensor data', color = 'r')
ax3.plot(temp, func(temp, *optimizedParameters2), label=fit_equation2, linestyle='--', linewidth=1, color='black')
ax3.legend()
plt.xlabel('Temperature ($^\circ$F)')
plt.ylabel('Resistance ($\Omega$)')
plt.title('Aftermarket 3 Wire Resistance vs Temperature Profile')
plt.grid(True, which = 'both')
plt.xticks(np.arange(60, 225, 10.0))
plt.xticks(rotation = 45)
plt.yscale("log")
plt.yticks([1000, 10000, 100000])
plt.subplots_adjust(top = .95, bottom = 0.15)
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(temp, stock_2_wire, label='Stock 2 Wire Sensor')
ax4.plot(temp, aftermarket_2_wire, label='Aftermarket 2 Wire Sensor', color = 'g')
ax4.plot(temp, aftermarket_3_wire, label='Aftermarket 3 Wire Sensor', color = 'r')
ax4.legend()
plt.xlabel('Temperature ($^\circ$F)')
plt.ylabel('Resistance ($\Omega$)')
plt.title('Temperature Sensor Resistance vs Temperature Comparison')
plt.grid(True, which = 'both')
plt.xticks(np.arange(60, 225, 10.0))
plt.xticks(rotation = 45)
plt.yscale("log")
plt.yticks([1000, 10000, 100000])
plt.subplots_adjust(top = .95, bottom = 0.15)
plt.show()

