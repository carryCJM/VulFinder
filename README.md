## Dataset


The original version of the dataset can be downloaded through the following link:
<a href="https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset">MSR_20_Code_vulnerability_CSV_Dataset </a>


We have provided the processed dataset information in `dataset/bigvul_all`
 

## Structure of the project
The project structure is as follows:
```
├── code2graph
│ ├── pdgc.py # The main PDG generation module, responsible for generating program dependency graphs
│ ├── joern  # Directory for storing Joern-related information and outputs
│ └── ... # Other related modules
├── main
│ ├── preprocess.py # Data preprocessing module, primarily used to generate hidden layer information of code using LLMs and save it
│ ├── run.py # # Entry point for the model training module, responsible for parsing command-line arguments and initiating the training process
│ └── run_utils.py  # Utility functions for training and evaluation, providing helper methods to streamline the process
│ ├── model.py # Defines the model architecture and related functionalities for the training process
│ └── ... # Other auxiliary modules
├── requirements.txt # Project dependencies, lists all required Python libraries to install
└── README.md  # Project documentation file, provides an overview and usage instructions
```

## How to Reproduce

### Environment Setup
First, install the required Python dependencies using the following command:
```
pip install -r requirements.txt
```

Next, download the Joern version and save it in the `code2graph/joern` folder.

### Code Execution

#### Joern Parsing
To parse the dataset, save the code into a single file and store the paths of all the code files in a text file. Then, execute the following commands:

```
cd code2graph
python pdgc.py
```
We have provided the processed dataset information in `dataset/bigvul_all`

#### Saving LLM Hidden Layer Information
To save the hidden layer information after processing the code with the LLM, execute the following command:

```
python main/preprocess.py
```
#### Model Training and Validation
Finally, to train and validate the model, run:

```
python run.py
```



