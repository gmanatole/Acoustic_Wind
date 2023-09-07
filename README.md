# Using underwater acoustics to estimate surface wind speed

This GUI contains all the tools necessary to get in-situ estimations of surface wind speed from an acoustic dataset.

## Estimate weather

This tab allows you to choose a model with which to estimate wind speed. The audio timestemp corresponds to the timestep between successive wind estimations. 
At each timestep, the model will look at a sample of length defined in the model's parameters (This length can be modified manually or automatically, see define user parameters).
Enter the desired batchsize, or how many samples the model will analyze at once. Enter default value of 8 if unsure.
After clicking on Estimate wind speed, a plot will show the wind speed estimation over time (plot assumes that wav files are subsequent). 
A table, that can be found as a csv file in the data directory, shows the wind speed and sound pressure level at each timestep.

## Download ERA wind speed

If you want to calibrate a model to your data and not use the parameters from the literature, ERA5 data is necessary. 
ERA5 is a reanalysis model that gives hourly wind speed data on a regular lat-lon grid of 0.25 degrees. https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
The UID and key of your CDS account can be found on your profile on the copernicus CDS website after account creation.
By default all the data for a selected day will be downloaded.

## Define user parameters

The tab allows the user to enter a model's parameters unknown to the toolbox and the hydrophone's parameters. 
This can also be done manually in the config directory by modifying the toml files. 
Be careful with the units of your data. 
For the model's parameters, only quadratic and logarithmic models are available for the moment.
Enter the parameters of the fitted model on the second line. For the quadratic model a*x**2 + b*x + c, parameters are entered as a,b,c. Check the empirical file in the models directory for the exact functions or to enter new models.
Enter the frequency, samplerate of analysis as well as the length of the signal (at each timestep) of analysis. This signal is then split into N subsignals with a chosen overlap.

## Calibrate model to data

Once the ERA5 data is downloaded, the model can be fitted to the acoustic data to yield the lowest error with ERA5 data as the ground truth. 
For each timestep, a weighted average in time and space of the ERA5 data joins the ground truth to the position and time of the hydrophone. 
With the final sound pressure level dataset and using skicit-learn's stratified K-Folds cross validation, the fitted parameters of each K Fold are averaged to obtain the final fit to the acoustic data.
In order to get the wind speed estimation, go back to the Estimate weather tab and select user_{model}.


### Computing the sound pressure level

The sound pressure level (one value at every timestep) is either computed in the estimate weather tab or the calibrate model to data tab. The method used is the one defined by Merchant. (See supporting informations in https://doi.org/10.1111/2041-210X.12330).
The calibration is as follow : $'S(f) = Mh(f) + G(f) + 20 \cdot log_{10}(\frac{1}{VADC}) + 20 \cdot log_{10}(2^{Nbit−1})'$




