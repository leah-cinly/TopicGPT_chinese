# TopicGPT_chinese
 This is a TopicGPT method adapted for Chinese topic identification.
 
 ### 工作说明：
 - 复现topicGPT，并将适用于英文文本挖掘的topicGPT改为中文文本适用。
 - 增加可选模型：国内LLMs的API调用，比如通义千问等。
 - 五个prompt文件内容撰写，所有文件中打开prompt文件的代码部分需要添加“encoding=utf-8”。
 - 所有topic_pattern匹配topics时的正则表达式需换成：
 topic_format = regex.compile(r"^\[(\d+)\] ([^:：]+)[：:](.+)$")。
 - Config.yaml配置文件的修改。
 - 解决输出的jsonl格式文件出现转义字符的问题，使其显示中文输出。
 - 其它很多函数的debug以及错误机制处理等完善。
 
 
 ## 以下是有关TopicGPT原项目声明
 [![arXiV](https://img.shields.io/badge/arxiv-link-red)](https://arxiv.org/abs/2311.01449) [![Website](https://img.shields.io/badge/website-link-purple)](https://chtmp223.github.io/topicGPT) 
 
 This repository contains scripts and prompts for our paper ["TopicGPT: Topic Modeling by Prompting Large Language Models"](https://arxiv.org/abs/2311.01449) (NAACL'24). Our `topicgpt_python` package consists of five main functions: 
 - `generate_topic_lvl1` generates high-level and generalizable topics. 
 - `generate_topic_lvl2` generates low-level and specific topics to each high-level topic.
 - `refine_topics` refines the generated topics by merging similar topics and removing irrelevant topics.
 - `assign_topics` assigns the generated topics to the input text, along with a quote that supports the assignment.
 - `correct_topics` corrects the generated topics by reprompting the model so that the final topic assignment is grounded in the topic list. 
 
 ## 📦 Using TopicGPT
 ### Getting Started
 1. Make a new Python 3.9+ environment using virtualenv or conda. 
 2. Install the required packages:
     ```
     pip install topicgpt_python
     ```
 - Set your API key:
     ```
     # Run in shell
     # Needed only for the OpenAI API deployment
     export OPENAI_API_KEY={your_openai_api_key}
 
     # Needed only for the Vertex AI deployment
     export VERTEX_PROJECT={your_vertex_project}   # e.g. my-project
     export VERTEX_LOCATION={your_vertex_location} # e.g. us-central1
 
     # Needed only for Gemini deployment
     export GEMINI_API_KEY={your_gemini_api_key}
 
     # Needed only for the Azure API deployment
     export AZURE_OPENAI_API_KEY={your_azure_api_key}
     export AZURE_OPENAI_ENDPOINT={your_azure_endpoint}
     ```
 - Refer to https://openai.com/pricing/ for OpenAI API pricing or to https://cloud.google.com/vertex-ai/pricing for Vertex API pricing. 
 

 ## 📜 Citation
 ```
 @misc{pham2023topicgpt,
       title={TopicGPT: A Prompt-based Topic Modeling Framework}, 
       author={Chau Minh Pham and Alexander Hoyle and Simeng Sun and Mohit Iyyer},
       year={2023},
       eprint={2311.01449},
       archivePrefix={arXiv},
       primaryClass={cs.CL}
 }
 ```
