require(fpp)

#Answer 1:
plot(elecequip)

#Answer 2:
ans = stl(elecequip, s.window = 'periodic')

#Answer 3: Yes, the data is seasonal.
plot(seasadj(ans))

#Answer 4: There is no need to do variance stablization as seen from the data plot. 
#Plotting the data after BoxCox transformation, it is quite similar to the original seasonal adjusted data
#lambda = BoxCox.lambda(elecSeasAdj)
#elecVar = BoxCox(elecSeasAdj, lambda)
#plot(elecVar)

#Answer 5: The data is not stationary as Acf decreases slowly.
#We can check stationarity using adf.test since we already have removed seasonality.
Acf(seasadj(ans))
adf.test(seasadj(ans))


#Answer 6: Yes, after differencing the non-stationary data has been converted to stationary
#Acf now decreases to zero quite fast
nd = ndiffs(seasadj(ans))
elecDiff = diff(seasadj(ans), differences = nd)
Acf(elecDiff)
adf.test(elecDiff)

#Answer 7: The model returns the following values: p = 3, d = 0, q = 1
model = auto.arima(elecDiff)
model$aicc

#Answer 8: AICc for: ARIMA(4,0,0) = 981.36, ARIMA(3,0,0) = 981.66, ARIMA(2,0,0) = 998.87
# Best Value is for original auto arima model
model1 = Arima(elecDiff, order=c(4,0,0))
model1$aicc
model2 = Arima(elecDiff, order=c(3,0,0))
model2$aicc
model3 = Arima(elecDiff, order=c(2,0,0))
model3$aicc

bestmodel = model

#Answer 9: Acf suggests that the residuals are white noise
#From Box.test() we find p-val is close to 1 which suggests that the null hypothesis can't be rejected.
#So, the residuals are white noise
errors = residuals(bestmodel)
Acf(errors)
Box.test(errors)

#Answer 10: The model is proper. Forecast is plotted below
plot(forecast(bestmodel))