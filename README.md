# Transformer Character Counter API

This repository contains an implementation of the Transformer model for the task of predicting the number of times a character has appeared previously in a string.

# Task Description
The task involves predicting, for each position in a string, how many times the character at that position has occurred previously, with a maximum count of 2. The labels are classified into three classes: 0, 1, or >2, representing the number of occurrences. For example:

```
String: one two three four f
Labels: 00000011100122020121
```

## Data
The repository includes two files: train.txt and test.txt, both under the data folder. Each file contains strings with a length of 20 characters per line. The label for each string is generated using the provided `count_letters` function in the utils.py file.

## Implementation Details
- The model architecture is based on the Transformer model.
- The vocabulary consists of 27 character types (lowercase letters and space).
- The model receives strings of length 20.
- The labels are generated dynamically at runtime.
- The model is served through a REST API using the FastAPI framework.

## Requirements
- Python (created and tested with version 3.10.5)
- Poetry (created and tested with version 1.6.1)
- (Optional) Docker

## Setup
1. Start by cloning the repository into your local environment.
2. Install poetry in your local environment by running: pip install poetry
3. Create the virtual environment for the project by running: poetry install
4. Initialize the virtual environment by running: poetry shell
5. Run the entrypoint script with: python main.py

## Usage

1. **Training**: Run `main.py` to train the Transformer model using the provided data.
2. **API Server**: Ensure that the FastAPI server is started by running `main.py.`
3. **Testing**: Use the Swagger UI available at http://localhost:8000/docs to test the API endpoint, or use tools such as curl or Postman.

## Endpoint

The API endpoint accepts a JSON object with the structure:
```
{
    "text": "one two three four f"
}
```
And returns a JSON object with the structure:
```
{
  "prediction": "00000011100122020121"
}
```
Where prediction is the string of labels predicted by the trained model.


