## AI2THOR Data Preparation

Look at `globals_dirs.py` and change the folder paths where you would like to store the data


### Download pre-generated data
We provide preprocessed RGB-D data (~45G) for all scenes. You can download it using gdown in the data directory.

```bash
gdown --id 1H9fiGZg8ILSOhfssRxF1rXOVKyfFwepa
```

### Generating Data Yourself
You can skip this step if you downloaded the pre-generated data

Start the ai2thor simulator by running the following command
```bash
python data_preparation/ai2thor/startx.py
```
Note: This is tested on headless remote servers, and ran inside a GPU node.

On a seperate terminal, execute the following commands on the same GPU where you started the simulator

```bash
python data_preparation/ai2thor/ai2thor_datagen.py
```
Remember to change the `DATA_DIR` variable to wherever you want to store the data

### Generating the split files
Next, run the following command to get the coco format json we use
```bash
python data_preparation/ai2thor/ai2thor2coco.py
```
