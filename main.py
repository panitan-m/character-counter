import numpy as np
import os
import uvicorn
import torch

from nlp_engineer_assignment import count_letters, print_line, read_inputs, \
    score, train_classifier


def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']

    ###
    # Train
    ###

    train_inputs = read_inputs(
        os.path.join(cur_dir, "data", "train.txt")
    )

    model = train_classifier(train_inputs, vocabs)
    os.makedirs('./save_model', exist_ok=True)
    torch.save(model, 'save_model/model.pt')

    ###
    # Test
    ###

    test_inputs = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )

    # TODO: Extract predictions from the model and save it to a
    # variable called `predictions`. Observe the shape of the
    # example random predictions.
    
    # Character to index dictionary
    char_to_index = get_char_to_index()
    
    test_input_ids = torch.tensor(list(map(lambda x: [char_to_index[char] for char in x], test_inputs)))
    
    golds = np.stack([count_letters(text) for text in test_inputs])
    
    test_batch_size = 100
    num_test_inputs = len(test_inputs)
    num_batches = num_test_inputs // test_batch_size
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            test_batch = test_input_ids[i*test_batch_size:(i+1)*test_batch_size]
            _, prediction_batch = model(test_batch)
            predictions.append(prediction_batch)
    predictions = np.concatenate(predictions, axis=0)

    # Print the first five inputs, golds, and predictions for analysis
    for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(
            f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}"
        )
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

    print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
    print_line()
    
    
def get_char_to_index():
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    char_to_index = {char: index for index, char in enumerate(vocabs)}
    return char_to_index


def predict_letter_occurrence(model, char_to_index, text):
    input_ids = torch.tensor([char_to_index[char] for char in text]).unsqueeze(0)
    _, prediction = model(input_ids)
    prediction = prediction.squeeze(0)
    return prediction


if __name__ == "__main__":
    train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )
