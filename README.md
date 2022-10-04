# 1. Description

The task is to determine whether the ad has contact information

## 1.1 Dataset
There are the following fields for training and inference:
* `title`
* `description`
* `subcategory`
* `category`
* `price`
* `region`
* `city` 
* `datetime_submitted`

Target - `is_bad`.

There are two datasets: `train.csv` и `val.csv`. 
In datasets there may be (as, unfortunately, in any marked data) incorrect labels.

The `train.csv` contains more data, but its markup is less accurate.

In `val.csv` there is significantly less data, but the markup is more accurate.

The test dataset on which the solution is evaluated will be more like `val.csv`.

The `val.csv` is in the `./data` folder. 
The `train.csv` script can be downloaded by `./data/get_train_data.sh` or by going to 
[link](https://drive.google.com/file/d/1LpjC4pNCUH51U_QuEA-I1oY6dYjfb7AL/view?usp=sharing) 

## 1.2 Task
In the problem you need to estimate the probability of the presence of contact information in the ad. 
The result of the model is an `pd.DataFrame` with columns:
* `index`: `int`, position of the record in the file;
* `prediction`: `float` from 0 to 1.

Example:

|index  |prediction|
|-------|----------|
|0|0.12|
|1|0.95|
|...|...|
|N|0.68|

The average `ROC-AUC` for each ad category will be used as a metric for the quality of the model's performance.


# 2. Launching the solution

The code for training and inference models must be located in the `./lib` folder. 

In order for us to test your solution, we need to change the `process` method of the `Test` class in the `./lib/run.py` file. 

This is where the inference of your model on the test data takes place. 

The method should return two dataframes with answers to problems 1 and 2, respectively.

You can access validation, train (if the file is downloaded) and test data using the 'val', 'train' and 'test' methods.


To make it easier to understand how models are run, we have prepared constant 
"models" (`./lib/model.py`), which are tried in `./lib/run.py` to form the final answer.

The formats of the test file (it will lack the `is_bad` stack), the responses of the exercises 1 and 2 are given above. 
After the run, minimal checks will be run for answers to the described format

The solution will be checked automatically. 
Before submitting your solution, you will need to make sure that everything is working correctly by running the command 
`docker-compose -f docker-compose.yaml up` in the root of this repository. 
All local repository code is copied to the `/app` folder in the container, and the local `./data` folder is copied to `/data` in the container.
After that, run the `python lib/run.py --debug` command.
You must have `docker` and `docker-compose` installed on your system to make it work.

You can add the required libraries to the `requirements.txt` file or directly to the `Dockerfile`.

The container will not have access to the internet during the model inference. 

Note that the container uses python3.8 by default.


# 3. Resources

Container resources:
* 4 GB RAM
* 2 CPU cores
* 1 GPU, 2GB memory

Run time limit:
* 60,000 objects must be processed for no more than 180 minutes for the pre-saved model on the CPU and 30 minutes on the GPU.

**It's important that everything you need to run.py is in the repository.**\
Often solvers offer to download the archive with weights model before starting manually, in this case it is necessary that the weights are downloaded and unpacked at assembly of the container or training takes place in the pipeline.

# Baseline

The current baseline to beat for the first part is 0.92.

# Model result

Categories:
* Бытовая электроника: 0.9608737099399856
* Недвижимость: 0.9866777893650743
* Транспорт: 0.9911951655525614
* Работа: 0.9574465659298896
* Для дома и дачи: 0.9642454583088893
* Услуги: 0.9387760412884234
* Личные вещи: 0.9571000991233776
* Для бизнеса: 0.9477649786278346
* Хобби и отдых: 0.9310013324113272
* Животные: 0.9763405220349332

The average ROC AUC: 0.9611
