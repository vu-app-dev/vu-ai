# VU

This is minimal implementation of the AI Interview System.

## Requirements

- python 3.13 or later

#### Install python using MiniConda

1) Downlaod and install MiniConda from [here]([text](https://www.anaconda.com/docs/getting-started/miniconda/install))
2) Create a new environment using the following command:
```bash
$ conda create -n vu-app
```

3) Activate the environment
```bash
$ conda activate vu-app
```

## Installation

### Install the required packages

```bash
$pip install -r requirements.txt
```

### setup the environment variables

```bash
$ cp .env.example .env
```

set your environment variables in the `.env` file. Like `OPENAI_API_KEY` value.
