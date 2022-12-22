# ssf

## Overview  
The project is mainly for the Cavity ring-down spectroscopy (CRDS) with the comb laser.  
The large scale of crds data caused difficulty in analyzing and storing the data.  
Therefore, We plan to design a filter to filter out some redundant data and precisely pinpoint the useful data.  


In the Spectrum Signal Filter (SSF), the output of SSF finally created the suggested data (./output/*.txt) each of which includes one peak of crds data.  
In the meanwhile, SSF also marked the order of comb's lines in different scanning(pzt) by .txt name  


### First step:

```cpp
mkdir raw_data
```

copy a raw data to ./raw_data/

### Second step:

```cpp
python do_simulator.py -i "raw_data name"
```
ex: 
python do_simulator.py -i "sample.txt"

### Third step:

open do_peak_sensor.ipynb  

In the function, it would play a sensing role to directly find the peak's locations  
In the ipynb, you also can check the sensing result
![image](https://user-images.githubusercontent.com/64359495/209142649-7d915dd6-b24a-486e-bb70-2de42610983f.png)

Save the result to './peak_sensor/'

### Last step:

open main.ipynb  

Now, it's still programming....

