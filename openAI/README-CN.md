[English](README.md) | [中文](README-CN.md)

# ChatGPT API Python 示例

这是一个使用OpenAI的ChatGPT API生成响应的Python示例代码。 代码使用`openai`包与API交互。

## 要求

* Python 3.6或更高版本
* `openai`包（您可以通过pip安装：`pip install openai`）

## 用法

1. 将代码中的`OPENAI_API_KEY`变量的值设置为您的OpenAI API密钥。
2. 实例化`Template`类并调用其`get_response`方法，以生成响应。 或者，您可以使用`get_response_stream`方法在流模式下生成响应。
3. 生成的响应将打印到控制台上。

## 代码解释

* `OPENAI_API_KEY`变量保存与OpenAI API交互所需的API密钥。
* `get_response`方法使用ChatGPT API生成对用户消息的响应。 使用`openai.ChatCompletion.create`方法创建新的聊天完成，将`model`参数设置为要使用的GPT-3模型的名称，将`messages`参数设置为包含用户消息的列表。
* `get_response_stream`方法使用流模式生成用户消息的响应。 它将用户消息作为输入，使用`openai.ChatCompletion.create`方法创建新的聊天完成，将`stream`参数设置为`True`，并遍历API返回的响应块，以生成最终响应。

## 示例

输入：“您好，您能帮我写一些代码吗？”

输出：“当然，我很乐意帮忙。您具体想做什么？”
