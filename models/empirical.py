import numpy as np

get = {
	   'quadratic' : lambda x, a, b, c, d : a*(x-d)**2+b*(x-d)+c,
	   'logarithmic' : lambda x, freq, a, b, c : 10**((x-a+10*b*np.log10(int(freq)))/(20*c))   
	   }
