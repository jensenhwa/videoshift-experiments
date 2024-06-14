# VideoShift Experiments
VideoShift is a benchmark for assessing distribution shifts in video data, with a focus on recognizing activities occuring in both domestic and professional settings.

## Quickstart

`main.sh`: Train VideoCLIP model.

`main2.sh`: Train Frozen-In-Time model on HOMAGE and evaluate on WAGE.

## Usage

```console
$ python3 main.py -h 
usage: main.py [-h] [-d DATASET [DATASET ...]] [-d_test DATASET_TEST]
               [-s N_SHOTS [N_SHOTS ...]] [-w N_WAY [N_WAY ...]]
               [--n_episodes N_EPISODES] [--val_tuning VAL_TUNING]
               [--class_split] [-m {gp,forest,random,grid}] [-n N_SEARCH_RUNS]
               [-f FOLDER] [--unfreeze_head] [--learning_rate LEARNING_RATE]
               [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--top_k TOP_K]
               [--label_verb]
               {clip,miles,videoclip,fit}
               {vl_proto,hard_prompt_vl_proto,nearest_neighbor,gaussian_proto,linear,subvideo,tip_adapter,coop,cona,cona_adapter,cona_prompt_init,name_tuning,name_tuning_adapter,coop_adapter}

positional arguments:
  {videoclip,fit}
                        VLM to run. Requires corresponding conda environment
  {vl_proto,hard_prompt_vl_proto,nearest_neighbor,gaussian_proto,linear,subvideo,tip_adapter,coop,cona,cona_adapter,cona_prompt_init,name_tuning,name_tuning_adapter,coop_adapter}
                        Classifier to run. Currently, only `linear` is supported.

options:
  -h, --help            show this help message and exit
  -d DATASET [DATASET ...], --dataset DATASET [DATASET ...]
                        Which dataset name to run on
  -d_test DATASET_TEST, --dataset_test DATASET_TEST
                        Which dataset name to test on
  -s N_SHOTS [N_SHOTS ...], --n_shots N_SHOTS [N_SHOTS ...]
                        Number of shots to run on
  -w N_WAY [N_WAY ...], --n_way N_WAY [N_WAY ...]
                        Number of categories to classify between. Default
                        value None indicates choosing the max categories for
                        each dataset.
  --n_episodes N_EPISODES
                        Number of support set samples to repeat every test
                        over
  --val_tuning VAL_TUNING
                        Whether or not the final trained classifier is
                        reloaded from the epoch with the best val performance
  --class_split         Flag to use class-wise splitting (meta-learning
                        paradigm) instead of video-wise splitting (finetuning
                        paradigm)
  -m {gp,forest,random,grid}, --method {gp,forest,random,grid}
                        Hyperparameter search method name.
  -n N_SEARCH_RUNS, --n_search_runs N_SEARCH_RUNS
                        Sets the max number of hyperparameter search runs per
                        test parameter value (dataset+n_shot)
  -f FOLDER, --folder FOLDER
                        Optional folder path in which to save val and test
                        results. By default creates folder for VLM and
                        Classifier choice
  --unfreeze_head       Flag to unfreeze and train head
  --learning_rate LEARNING_RATE
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --top_k TOP_K
  --label_verb          If true, only verb part of label is used
```
