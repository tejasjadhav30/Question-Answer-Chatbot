from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the models and tokenizers
encoder_model = load_model('encoder_model.keras')
decoder_model = load_model('decoder_model.keras')

with open('source_tokenizer.pkl', 'rb') as f:
    source_tokenizer = pickle.load(f)

with open('target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

# Constants
max_source_seq_length = 15

def decode_sequence(input_seq):
    # Encode the input sequence to get the states
    states_value = encoder_model.predict(input_seq)

    # Generate an empty target sequence with only the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<sos>']  # Assuming '<sos>' is the start token

    # Store the decoded sentence
    decoded_sentence = ''
    stop_condition = False

    while not stop_condition:
        # Predict the next word
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the index of the most likely word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, '')

        

        # Exit condition: stop token or max length
        if (sampled_word == '<eos>' or len(decoded_sentence.split()) > 50):
            stop_condition = True

        # Append the word to the decoded sentence
        if (sampled_word != '<eos>') :
            decoded_sentence += ' ' + sampled_word
        
        # Update the target sequence and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()

def preprocess_input(sentence):
    # Tokenize the input sentence
    seq = source_tokenizer.texts_to_sequences([sentence])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_source_seq_length, padding='post')
    return padded_seq

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Parse user input from the request
    user_input = request.form.get('message', '')

    if not user_input:
        return jsonify({'response': 'Please enter a message!'})

    # Preprocess the input
    input_seq = preprocess_input(user_input)

    # Generate the response
    response = decode_sequence(input_seq)
    return jsonify({'response': response})

if __name__ == '_main_':
    app.run(debug=True)