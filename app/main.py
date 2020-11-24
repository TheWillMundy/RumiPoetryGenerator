from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn

# ML Imports
import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

## Set Default Values
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
DEFAULT_SEED = 42
START_TOKEN = "<BOS>"
STOP_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
special_tokens_dict = {'bos_token': START_TOKEN, 'eos_token': STOP_TOKEN, 'pad_token': PAD_TOKEN}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# So we can use these later
async def setup_model():
    ### Setup ML model
    ## Initialize model
    model_class = GPT2LMHeadModel
    tokenizer_class = GPT2Tokenizer

    # Initialize tokenizer with special tokens
    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./model")
    tokenizer.add_special_tokens(special_tokens_dict)

    # Initialize model from pretrained Rumi model
    global model
    model = GPT2LMHeadModel.from_pretrained("./model")
    model.to(device)

middleware = [
    Middleware(CORSMiddleware, allow_origins=['*'])
]

app = Starlette(debug=True, middleware=middleware, on_startup=[setup_model])

### Setup ML model
## Set Default Values
# MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
# DEFAULT_SEED = 42
# START_TOKEN = "<BOS>"
# STOP_TOKEN = "<EOS>"
# PAD_TOKEN = "<PAD>"
# special_tokens_dict = {'bos_token': START_TOKEN, 'eos_token': STOP_TOKEN, 'pad_token': PAD_TOKEN}

# ## Initialize model
# model_class = GPT2LMHeadModel
# tokenizer_class = GPT2Tokenizer

# # Initialize tokenizer with special tokens
# tokenizer = GPT2Tokenizer.from_pretrained("./model")
# tokenizer.add_special_tokens(special_tokens_dict)

# # Initialize model from pretrained Rumi model
# model = GPT2LMHeadModel.from_pretrained("./model")
# device = torch.device("cpu")
# model.to(device)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

@app.route('/generate')
async def generate(request):
    """
    Using the model, generates a sample poem from Rumi
    """
    # Parse the input requests
    length = int(request.query_params['length']) if request.query_params['length'] else 300
    prompt_text = request.query_params['prompt'] if request.query_params['prompt'] else ""
    prefix = "<BOS> "
    encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=True, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    # Run the poetry generator here
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length + len(encoded_prompt[0]),
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
    )

    sequence = ""
    for idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(STOP_TOKEN)]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        total_sequence = total_sequence.replace(START_TOKEN, "\n")
        total_sequence = total_sequence.replace(PAD_TOKEN, "\n")

        sequence = total_sequence
    
    response = []
    response.append(sequence)

    # Return the output
    return JSONResponse({ 'poem': sequence })

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
