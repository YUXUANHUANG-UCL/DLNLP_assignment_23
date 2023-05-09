# DLNLP_assignment_23

## How to run the code for providing accurate feedback on the language proficiency of English Language Learners (ELLs)?

1. cd to the DLNLP_assignment_23 path:
```
cd your/file/path/to/DLNLP_assignment_23
```

2. create the necessary environment:
```
conda env create -f environment.yml
```

3. activate conda environment
```
conda activate tf_gpu
```

4. start the code:

```
python main.py
```

## Role of each folder and file

**Datasets** contains all datasets and they are stored in a structure.

**figure** stores result images.

**lstm_lstm** stores model of LSTMs-LSTMs.

**lstm_trans** stores model of LSTMs-Transformer Encoder.

**trans_lstm** stores model of Transformer Encoder-LSTMs.

**trans_trans** stores model of Transformer Encoder-Transformer Encoder.

**environment.yml** stores dependent packages and other environment information.

**main.py** is the file to start the whole process of the assignment.

**prediction.py** includes functions to predict scores and evaluate the prediction results.

**visualisation.py** includes functions to load models to predict high-resolution images from low-resolution images and visulise results.
