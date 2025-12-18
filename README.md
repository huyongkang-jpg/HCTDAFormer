# HCTDAFormer
This is the pytorch implementation of HCTDAFormer. I hope these codes are helpful to you!

## Requirements
The code is built based on Python 3.9.12, PyTorch 2.1.0, and NumPy 1.21.2，pandas 1.3.5，scipy 1.7.3.

## Datasets
You can directly use the relevant datasets provided in our `data` folder. Alternatively, you can obtain the original datasets from their official repositories:
* **STSGCN** (PEMS03, PEMS04, PEMS07, PEMS08): [GitHub Link](https://github.com/Davidham3/STSGCN)
* **ESG** (NYCBike, NYCTaxi): [GitHub Link](https://github.com/LiuZH-19/ESG)
* **LargeST** (GBA, GLA, CA): [GitHub Link](https://github.com/liuxu77/LargeST)

We provide preprocessed datasets [here](https://drive.google.com/drive/folders/1-5hKD4hKd0eRdagm4MBW1g5kjH5qgmHR?usp=sharing). We express our sincere gratitude to the providers of all the aforementioned datasets.


## Train Commands
It's easy to run! Here are some examples, and you can customize the model settings in train.py.
### PEMS08
```
nohup python -u train.py --data PEMS08 > PEMS08.log &
```
### NYCBike Drop-off
```
nohup python -u train.py --data bike_drop > bike_drop.log &
```