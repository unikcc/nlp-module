[English](README.md) | [中文](README-CN.md)

# ChatGPT API Python Example

This is a Python example code that demonstrates how to use OpenAI's ChatGPT API to generate responses to user messages. The code uses the `openai` package to interact with the API.

## Requirements

* Python 3.6 or later
* `openai` package (you can install it via pip: `pip install openai`)

## Usage

1. Set the value of the `OPENAI_API_KEY` variable in the code to your OpenAI API key.
2. Instantiate the `Template` class and call its `get_response` method with a user message to generate a response. Alternatively, you can use the `get_response_stream` method to generate a response in a streaming mode.
3. The generated response will be printed to the console.

## Code explanation

* The `OPENAI_API_KEY` variable holds the API key required to interact with the OpenAI API.
* The `get_response` method generates a response to a user message using the ChatGPT API. The `openai.ChatCompletion.create` method is used to create a new chat completion, with the `model` parameter set to the name of the GPT-3 model to use, and the `messages` parameter set to a list containing the user message.
* The `get_response_stream` method generates a response to a user message using the ChatGPT API in a streaming mode. It takes a user message as an input, creates a new chat completion using the `openai.ChatCompletion.create` method with the `stream` parameter set to `True`, and iterates through the chunks of responses returned by the API to generate the final response.

## Example

Input: "Hello, can you help me with some coding?"

Output: "Sure, I'd be happy to help. What specifically are you trying to do?"
