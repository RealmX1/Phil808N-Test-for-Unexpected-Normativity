create conda environment with python 3.12
```bash
conda create -n llm_unexpected_normativity python=3.12
```

activate the environment
```bash
conda activate llm_unexpected_normativity
```

install the dependencies
```bash
pip install -r requirements.txt
```

run the main.py
```bash
python main.py --mode generate
python main.py --mode annotate
```
The evaluate and visualize mode are not fully functional yet
