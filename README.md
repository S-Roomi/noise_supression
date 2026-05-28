# Noise Suppression
Learning more and design own Noise Suppression Model

## Requirments / Tools
project uses [uv](https://docs.astral.sh/uv/)

To sync up the depenedencies, ensure the pyproject.toml is pulled and run
```bash
uv sync
```

## DataSet
Clean speach from [LibriSpeech](https://www.openslr.org/resources/12) 
[data](https://www.openslr.org/resources/12/dev-clean.tar.gz)

Noise data from [ESC-50] dataset
[data](https://github.com/karoldvl/ESC-50/archive/master.zip)

Get dataset with the following commands
```bash
mkdir -p data
cd data
wget https://www.openslr.org/resources/12/dev-clean.tar.gz && wget https://github.com/karoldvl/ESC-50/archive/master.zip
```


## Todo
- [x] Create mixing script for clean and noisy datasets
- []
- []

## References

