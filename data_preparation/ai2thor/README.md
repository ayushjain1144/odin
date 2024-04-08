## AI2THOR Data Preparation

Look at `globals_dirs.py` and change the folder paths where you would like to store the data

Start the ai2thor simulator by running the following command
```bash
python data_preparation/ai2thor/startx.py
```

On a seperate terminal, execute the following commands on the same GPU where you started the simulator

```bash
python data_preparation/ai2thor/ai2thor_datagen.py
```
Remember to change the `DATA_DIR` variable to wherever you want to store the data

Next, run the following command to get the coco format json we use
```bash
python data_preparation/ai2thor/ai2thor2coco.py
```