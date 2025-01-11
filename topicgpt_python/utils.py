import os
import regex
import json
import time
import pandas as pd
from anytree import Node
import traceback
import subprocess

from openai import OpenAI, AzureOpenAI
import tiktoken
from vllm import LLM, SamplingParams
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from anthropic import AnthropicVertex
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from sklearn import metrics
import numpy as np


class APIClient:
    """
    Prompting for OpenAI, VertexAI, and vLLM.

    Parameters:
    - api: API type (e.g., 'openai', 'vertex', 'vllm','aliyun')
    - model: Model name

    Methods:
    - estimate_token_count: Estimate token count for a prompt
    - truncating: Truncate document to max tokens
    - iterative_prompt: Prompt API one by one with retries
    - batch_prompt: Batch prompting for vLLM API
    """

    def __init__(self, api, model):
        self.api = api
        self.model = model
        self.client = None

        # Setting API key ----
        if api == "openai":
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif api == "vertex":
            vertexai.init(
                project=os.environ["VERTEX_PROJECT"],
                location=os.environ["VERTEX_LOCATION"],
            )
            if model.startswith("gemini"): 
                self.model_obj = genai.GenerativeModel(self.model)
        elif api == "vllm":
            self.hf_token = os.environ.get("HF_TOKEN")
            self.llm = LLM(
                self.model,
                download_dir=os.environ.get("HF_HOME", None),
            )
            self.tokenizer = self.llm.get_tokenizer()
        elif api == "gemini": 
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model_obj = genai.GenerativeModel(self.model)
        elif api == "azure": 
            self.client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2024-02-01",
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
        elif api == "aliyun":
            self.client=OpenAI(
            api_key=os.getenv("ALIYUN_OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            raise ValueError(
                f"API {api} not supported. Custom implementation required."
            )

    def estimate_token_count(self, prompt: str) -> int:
        """
        Estimating the token count for the prompt with tiktoken

        Parameters:
        - prompt: Prompt text

        Returns:
        - token_count: Estimated token
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")

        token_count = len(enc.encode(prompt))
        return token_count

    def truncating(self, document: str, max_tokens: int) -> str:
        """
        Truncating the document to the max tokens

        Parameters:
        - document: Document text
        - max_tokens: Maximum token count

        Returns:
        - truncated_doc: Truncated document
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            enc = tiktoken.get_encoding("o200k_base")

        tokens = enc.encode(document)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return enc.decode(tokens)

    def iterative_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,  # default value for top_p in openai
        system_message: str = "You are a helpful assistant.",
        num_try: int = 3,
        verbose: bool = False,
    ):
        """
        Prompting API one by one with retries

        Parameters:
        - prompt: Prompt text
        - max_tokens: Maximum token count
        - temperature: Temperature for sampling
        - top_p: Top p value for sampling
        - system_message: System message
        - num_try: Number of retries
        - verbose: Verbose mode

        Returns:
        - response: Response text
        """
        # Formatting prompt
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(num_try):
            try:
                if self.api in ["openai", "azure","aliyun"]:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    if verbose:
                        print(
                            "Prompt token usage:",
                            completion.usage.prompt_tokens,
                            f"~${completion.usage.prompt_tokens/1000000*5}",
                        )
                        print(
                            "Response token usage:",
                            completion.usage.completion_tokens,
                            f"~${completion.usage.completion_tokens/1000000*15}",
                        )
                    return completion.choices[0].message.content

                elif self.api == "vertex":
                    if self.model.startswith("claude"):
                        client = AnthropicVertex(
                            region=os.environ["VERTEX_LOCATION"],
                            project_id=os.environ["VERTEX_PROJECT"],
                        )
                        message = client.messages.create(
                            model=self.model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system_message,
                            messages=[message[1]],
                        )
                        message_json_str = message.model_dump_json(indent=2)
                        message_dict = json.loads(message_json_str)
                        text_content = message_dict["content"][0]["text"]
                        if verbose:
                            print(
                                "Prompt usage:",
                                message_dict["usage"]["input_tokens"],
                                f"${message_dict['usage']['input_tokens']/1000000*3}",
                            )
                            print(
                                "Prompt usage:",
                                message_dict["usage"]["output_tokens"],
                                f"${message_dict['usage']['output_tokens']/1000000*15}",
                            )
                        return text_content
                    else:
                        config = GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        # safety config
                        safety_config = [
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]

                        try:
                            response = self.model_obj.generate_content(
                                system_message
                                + prompt,  # Didn't find a way to add system message in the API
                                generation_config=config,
                                safety_settings=safety_config,
                            )
                            return response.text.strip()
                        except:  # Avoid rate limiting issues
                            traceback.print_exc()
                            time.sleep(60)


                elif self.api == "vllm":
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop_token_ids=[
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                        ],
                    )
                    final_prompt = self.tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    vllm_output = self.llm.generate([final_prompt], sampling_params)
                    return [output.outputs[0].text for output in vllm_output][0]
                
                elif self.api == "gemini":
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    self.model_obj = genai.GenerativeModel(self.model)
                    config = genai.types.GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    # safety config
                    safety_config = {
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                          HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                          HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                          }
                    try:
                        response = self.model_obj.generate_content(
                            system_message
                            + prompt,  # Didn't find a way to add system message in the API
                            generation_config=config,
                            safety_settings=safety_config,
                        )
                        return response.text.strip()
                    except:  # Avoid rate limiting issues
                        traceback.print_exc()
                        time.sleep(60)

            except Exception as e:
                print(f"Attempt {attempt + 1}/{num_try} failed: {e}")
                if attempt < num_try - 1:
                    time.sleep(60)  # avoid rate limiting issues
                else:
                    raise

    def batch_prompt(
        self,
        prompts: list,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        system_message: str = "You are a helpful assistant.",
    ):
        """
        Batch prompting for vLLM API

        Parameters:
        - prompts: List of prompts
        - max_tokens: Maximum token count
        - temperature: Temperature for sampling
        - top_p: Top p value for sampling
        - system_message: System message

        Returns:
        - responses: List of response texts
        """
        if self.api != "vllm":
            raise ValueError("Batch prompting not supported for this API.")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

        prompt_formatted = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        final_prompts = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in prompt_formatted
        ]
        outputs = self.llm.generate(final_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class TopicTree:
    def __init__(self, root_name="Topics"):
        self.root = Node(name=root_name, lvl=0, count=1, desc="Root topic", parent=None)
        self.level_nodes = {0: self.root}

    @staticmethod
    def node_to_str(node, count=True, desc=True):
        if not count and not desc:
            return f"[{node.lvl}] {node.name}"
        elif not count and desc:
            return f"[{node.lvl}] {node.name}: {node.desc}"
        elif count and not desc:
            return f"[{node.lvl}] {node.name} (Count: {node.count})"
        else:
            return f"[{node.lvl}] {node.name} (Count: {node.count}): {node.desc}"

    @staticmethod
    def from_topic_list(topic_src, from_file=False):
        tree = TopicTree()
        
        # If from_file is True, read the file, otherwise use the provided list directly
        if from_file:
            with open(topic_src, "r", encoding='utf-8') as f:
                topic_list = f.readlines()
        else:
            topic_list = topic_src  # Assume topic_src is already a list of topics
        
        # Filter empty topics
        topic_list = [topic for topic in topic_list if len(topic.strip()) > 0]
        
        pattern = regex.compile(r"^\[(\d+)\] (.+) \(Count: (\d+)\)\s?:(.+)?")

        for topic in topic_list:
            if not topic.strip():
                continue
            try:
                match = regex.match(pattern, topic.strip())
                if match:
                    lvl, label, count, desc = (
                        int(match.group(1)),
                        match.group(2).strip(),
                        int(match.group(3)),
                        match.group(4).strip() if match.group(4) else "",
                    )
                    #print(f"Adding topic: Level={lvl}, Label={label}")  # Debugging
                    tree._add_node(lvl, label, count, desc, tree.level_nodes.get(lvl - 1))
                else:
                    print(f"Regex match failed for topic: {topic.strip()}")
            except Exception as e:
                print(f"Error processing topic: {topic.strip()}")
                print(e)
                traceback.print_exc()

        return tree

    def from_seed_file(self, seed_file):
        tree = TopicTree()
        topic_list = open(seed_file, "r", encoding='utf-8').readlines() if seed_file else []
        topic_list = [topic for topic in topic_list if len(topic.strip()) > 0]
        pattern = regex.compile(r"^\[(\d+)\] ([^:：]+)[：:](.+)$")
        for topic in topic_list:
            if not topic.strip():
                continue
            try:
                #match = regex.match(pattern, topic.strip())
                #lvl, label = int(match.group(1)), match.group(2).strip()
                match = regex.match(pattern, topic.strip())
                lvl, label, desc = int(match.group(1)), match.group(2).strip(), match.group(3).strip()

                # Ensure there's a desc (even if it's empty)
                #desc = ""  # You can set a default description here if needed

                #print(f"Adding topic: Level={lvl}, Label={label}")  # Print the level and label for debugging

                tree._add_node(lvl, label, 1, desc, tree.level_nodes.get(lvl - 1))
            except Exception as e:
                print(f"Error processing topic: {topic.strip()}")
                print(e)
                traceback.print_exc()
        return tree

    def _add_node(self, lvl, label, count, desc, parent_node):
        if parent_node:
            existing = next((n for n in parent_node.children if n.name == label), None)
            if existing:
                existing.count += count
            else:
                new_node = Node(
                    name=label, lvl=lvl, count=count, desc=desc, parent=parent_node
                )
                parent_node.add_child(new_node)
                
                if lvl not in self.level_nodes:
                    self.level_nodes[lvl] = []
                self.level_nodes[lvl].append(new_node)
                #print(f"Added node: {label} at level {lvl} to parent {parent_node.name}")
        else:
            print(f"Root node added: {label}")
            new_node = Node(name=label, lvl=lvl, count=count, desc=desc)
            self.root.add_child(new_node)  # Assuming self.root is the root node of the tree
            if lvl not in self.level_nodes:
                self.level_nodes[lvl] = []
            self.level_nodes[lvl].append(new_node)

    def _remove_node_by_name_lvl(self, name, lvl):
        node = next(
            (n for n in self.root.descendants if n.name == name and n.lvl == lvl), None
        )
        if node:
            node.parent = None

    def to_prompt_view(self, desc=True):
        def traverse(node, result=""):
            if node.lvl > 0:
                result += (
                    "\t" * (node.lvl - 1)
                    + self.node_to_str(node, count=False, desc=False)
                    + "\n"
                )
            for child in node.children:
                result = traverse(child, result)
            return result

        return traverse(self.root)

    def find_duplicates(self, name, level):
        return [
            node for node in self.root.descendants
            if node.name.lower() == name.lower() and node.lvl == level
        ]

    def to_file(self, fname):
        with open(fname, "w", encoding='utf-8') as f:
            if not self.root.descendants:
                print("No topics to save!")
            for node in self.root.descendants:
                #print(f"Node: {node.name}, Level: {node.lvl}, Desc: {node.desc}")  # Debug print
                indentation = "    " * (node.lvl - 1)
                f.write(indentation + self.node_to_str(node, count=True, desc=True) + "\n")
                #print(f"Saving node: {node.name}")  # Debug print to track what's being saved

    def to_topic_list(self, desc=True, count=True):
        return [self.node_to_str(node, count, desc) for node in self.root.descendants]

    def get_root_descendants_name(self):
        return [node.name for node in self.root.descendants]

    def update_tree(self, original_topics, new_topic_name, new_topic_desc):
        total_count = 0
        parent_node = None
        nodes_to_merge = []

        for name, lvl in original_topics:
            duplicates = self.find_duplicates(name, lvl)
            nodes_to_merge.extend(duplicates)
            total_count += sum(node.count for node in duplicates)
            if duplicates and not parent_node:
                parent_node = duplicates[0].parent

        if parent_node is None:
            parent_node = self.root

        merged_topic_node = (
            next(
                (node for node in parent_node.children if node.name == new_topic_name),
                None,
            )
            if parent_node
            else None
        )

        if merged_topic_node:
            final_count = total_count
            if final_count <= total_count:
                merged_topic_node.count = final_count
                merged_topic_node.desc = new_topic_desc
        else:
            if total_count <= sum(node.count for node in nodes_to_merge):
                merged_topic_node = self._add_node(
                    lvl=parent_node.lvl + 1,
                    label=new_topic_name,
                    count=total_count,
                    desc=new_topic_desc,
                    parent_node=parent_node,
                )

        for node in nodes_to_merge:
            if node != merged_topic_node:
                self._remove_node_by_name_lvl(node.name, node.lvl)

        return self

class Node:
    def __init__(self, name, lvl, count=1, desc="", parent=None):
        self.name = name
        self.lvl = lvl
        self.count = count
        self.desc = desc
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    @property
    def descendants(self):
        # Return all descendants (children and their children recursively)
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.descendants)  # Recursively add children's descendants
        return result


def calculate_purity(true_col, pred_col, df):
    """
    Calculate harmonic purity between two set of clusterings

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - purity: Purity score
    - inverse_purity: Inverse purity score
    - harmonic_purity: Harmonic purity score
    """
    contingency_matrix = metrics.cluster.contingency_matrix(df[true_col], df[pred_col])
    precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
    recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    purity = (
        np.amax(precision, axis=0) * contingency_matrix.sum(axis=0)
    ).sum() / contingency_matrix.sum()
    inverse_purity = (
        np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    harmonic_purity = (
        np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    return (purity, inverse_purity, harmonic_purity)


def calculate_metrics(true_col, pred_col, df):
    """
    Calculate topic alignment between df1 and df2 (harmonic purity, ARI, NMI)

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - harmonic_purity: Harmonic purity score
    - ari: Adjusted Rand Index
    - mis: Normalized Mutual Information
    """
    _, _, harmonic_purity = calculate_purity(true_col, pred_col, df)
    ari = metrics.adjusted_rand_score(df[true_col], df[pred_col])
    mis = metrics.normalized_mutual_info_score(df[true_col], df[pred_col])
    return (harmonic_purity, ari, mis)
