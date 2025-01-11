import pandas as pd
import argparse
from tqdm import tqdm
from topicgpt_python.utils import *
import jsonlines

def construct_document(api_client, docs, context_len):
    """
    Construct a list of documents corresponding to the parent node to generate subtopics for.

    Parameters:
    - api_client: API client object
    - docs: List of documents
    - context_len: Maximum token length for the prompt

    Returns: List of documents to generate subtopics for
    """
    i = 0
    doc_str, doc_prompt = "", []
    while i < len(docs):
        token_count = api_client.estimate_token_count(docs[i])
        to_add = (
            f"Document {i+1}\n" + " ".join(docs[i].split("\n")) + "\n"
            if token_count < context_len // 5
            else f"Document {i+1}\n{api_client.truncating(docs[i], context_len // 5)}\n"
        )
        if token_count >= context_len // 5:
            print(f"Truncating {token_count} to {context_len//5}....")

        if api_client.estimate_token_count(doc_str + to_add) >= context_len:
            doc_prompt.append(doc_str)
            doc_str = ""
        doc_str += to_add
        if i + 1 == len(docs):
            doc_prompt.append(doc_str)
            break
        i += 1
    return doc_prompt


import re

def parse_document_topics(df, topics_list):

    # Regex pattern to match topics in the format: [1] Topic Name: Description
    pattern = regex.compile(r"^\[(\d+)\] ([^:：]+)[：:](.+)$")
    
    all_topics = []
    
    responses = (
        df["refined_responses"]
        if "refined_responses" in df.columns
        else df["responses"]
    )
    
    #print("responses:", responses)

    for line in responses:
        line_topics = []
        # Assuming each line might contain multiple topics separated by newlines
        for topic in line.split("\n"):  # Split by newline to get individual topics
            #print("Debug1:",topic)
            match = regex.match(pattern, topic.strip())  
            #match = regex.match(topic_format, t)
            #topics_num, topic_name, description = int(match[1]), groups[2].strip(), groups[3].strip()
            
            if match:
                #topic_num, topic_name, description = match.groups()
                topic_num, topic_name, description = int(match[1]), match[2].strip(), match[3].strip()
                formatted_topic = f"[{topic_num}] {topic_name}"  # Format topic in the required form
                #print("Debug2:",formatted_topic)
                #if formatted_topic in topics_list:  # Check if topic is in topics_list
                line_topics.append(formatted_topic)
                #print("Debug3:",formatted_topic)
        all_topics.append(line_topics if line_topics else ["None"])
    
    #print("all_topics:", all_topics)  # Print the parsed topics to inspect
    return all_topics


def filter_topics_by_count(all_nodes, df):
    """
    Filter topics based on document count threshold.

    Parameters:
    - all_nodes: List of all nodes in the topic tree
    - df: DataFrame containing the document responses

    Returns: List of nodes that meet the threshold
    """
    return [
        node for node in all_nodes if node.count > len(df) * 0.01 and not node.children
    ]


def retrieve_documents(df, topic):
    if 'topics' not in df.columns:
        print("Error: Column 'topics' not found in the dataframe.")
        return []
    
    # Check if 'text' or any alternative text column exists
    text_column = None
    if 'text' in df.columns:
        text_column = 'text'
    elif 'document_text' in df.columns:
        text_column = 'document_text'
    elif 'content' in df.columns:
        text_column = 'content'
    
    if text_column is None:
        print("Error: Column for text data not found.")
        return []

    # Filter the documents based on the topic
    return df[df["topics"].apply(lambda x: topic in x)][text_column].tolist()



def construct_prompt(
    gen_prompt, current_topic, relevant_docs, max_tokens, context_len, api_client
):
    """
    Construct prompt for document generation.

    Parameters:
    - gen_prompt: Generation prompt
    - current_topic: Current topic
    - relevant_docs: List of relevant documents
    - max_tokens: Maximum token length for the prompt
    - context_len: Maximum token length for the prompt
    - api_client: API client object

    Returns: List of documents to generate subtopics for"""
    doc_len = context_len - api_client.estimate_token_count(gen_prompt) - max_tokens
    return construct_document(api_client, relevant_docs, doc_len)


def parse_and_add_topics(result, current_topic, pattern, verbose, topics_root):
    """
    Parse output and add topics to the tree.

    Parameters:
    - result: Output from the model
    - current_topic: Current topic
    - pattern: Regex pattern to match the output
    - verbose: Enable verbose output
    - topics_root: Topic tree object

    Returns: List of topics and prompts
    """
    names, prompt_top = [], []
    add_node = False
    prev_node = None
    for line in result.strip().split("\n"):
        line = line.strip()
        match = regex.match(pattern, line)
        if match:
            if ":" in line and add_node:  # add the topic to the tre
                lvl, name, description = (
                    int(match.group(1)),
                    match.group(2).strip(),
                    match.group(5).strip(" :"),
                )
                clean_name = name
                clean_description = description
                names.append(clean_name)
                prompt_top.append(f"{clean_name} (Count: 0): {clean_description}")
                if verbose:
                    print(f"{clean_name} (Count: 0): {clean_description}")
                topics_root._add_node(2, clean_name, 1, clean_description, prev_node)
            else:  # check if the topic in this line already exists
                lvl, name = int(match.group(1)), match.group(2).strip()
                if len(topics_root.find_duplicates(name, lvl)) > 0:
                    add_node = True
                    prev_node = topics_root.find_duplicates(name, lvl)[0]
        #elif verbose:
            #print(f"Not a match: {line}")
    return names, prompt_top

def generate_topics(
    api_client,
    df,
    topics_root,
    gen_prompt,
    context_len,
    max_tokens,
    temperature,
    top_p,
    verbose,
    max_topic_num=50,
):
    res, docs = [], []
    main_pattern = regex.compile(
        r"^\[(\d+)\] ([^\(\)]+?)\(Documents?: ((?:\d+(?:-\d+)?(?:, )?)+)\)([:\-\w\s,.\n'\&]*?)?$"
    )
    
    for parent_topic in tqdm(filter_topics_by_count(topics_root.root.descendants, df)):
        current_topic = f"[{parent_topic.lvl}] {parent_topic.name}"
        if verbose:
            print("Current topic:", current_topic)

        # Ensure that the 'topics' and 'text' columns exist in the DataFrame
        if 'topics' not in df.columns or 'text' not in df.columns:
            print("Error: Expected columns 'topics' and 'text' not found in the dataframe.")
            continue

        relevant_docs = retrieve_documents(df, current_topic)

        # Handle case when no relevant documents are found
        if not relevant_docs:
            print(f"No relevant documents found for topic: {current_topic}")
            continue

        doc_prompt = construct_prompt(
            gen_prompt,
            current_topic,
            relevant_docs,
            max_tokens,
            context_len,
            api_client,
        )

        for doc in doc_prompt:
            try:
                prompt = gen_prompt.format(Topic=current_topic, Document=doc)
                result = api_client.iterative_prompt(
                    prompt,
                    max_tokens,
                    temperature,
                    top_p=top_p,
                    system_message="You are a helpful assistant.",
                )
                if verbose:
                    print("Subtopics:", result)

                names, prompt_top = parse_and_add_topics(
                    result, parent_topic, main_pattern, verbose, topics_root
                )
                res.append(result)
                docs.append(doc)
            except Exception as e:
                res.append("Error")
                if verbose:
                    traceback.print_exc()
            if verbose:
                print("--------------------------------------------------")
    return res, docs


def generate_topic_lvl2(
    api, model, seed_file, data, prompt_file, out_file, topic_file, verbose
):
    api_client = APIClient(api=api, model=model)
    max_tokens, temperature, top_p = 1000, 0.0, 1.0

    if verbose:
        print("-------------------")
        print("Initializing topic generation (lvl 2)...")
        print(f"Model: {model}")
        print(f"Data file: {data}")
        print(f"Prompt file: {prompt_file}")
        print(f"Seed file: {seed_file}")
        print(f"Output file: {out_file}")
        print(f"Topic file: {topic_file}")
        print("-------------------")

    # Load data
    df = pd.read_json(data, lines=True)
    generation_prompt = open(prompt_file , 'r', encoding='utf-8').read()
    #print("generation_prompt",generation_prompt)
    topics_root = TopicTree().from_topic_list(seed_file, from_file=True)
    topics_list = topics_root.to_topic_list(desc=True, count=True)
    #print(df.head())
    parsed_topics = parse_document_topics(df, topics_list)
    #print(parsed_topics)  # Debug output for checking the topics
    df["topics"] = parsed_topics
    #df = df["topics"].apply(lambda x: [] if x is None else x)
    print(df["topics"].head()) 
    # Filter documents
    #df = df[df["topics"].apply(lambda x: x != ["None"])].reset_index(drop=True)
    df = df[df["topics"].apply(lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(i, str) and i != "None" for i in x))].reset_index(drop=True)
    
    if verbose:
        print("Number of remaining documents for prompting:", len(df))
    
    if len(df) == 0:
        print("No valid documents found after filtering. Exiting topic generation.")
        return topics_root
    # Generate topics
    res, docs = generate_topics(
        api_client,
        df,
        topics_root,
        generation_prompt,
        128000,
        max_tokens,
        temperature,
        top_p,
        verbose,
    )

    # Write results
    topics_root.to_file(topic_file)
    #pd.DataFrame({"text": docs, "topics": res}).to_json(
    #    out_file, orient="records", lines=True
    #)
    df = pd.DataFrame({"text": docs, "topics": res})
    with jsonlines.open(out_file, mode='w') as writer:
        for _, row in df.iterrows():
            writer.write(row.to_dict())
    return topics_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api", type=str, help="API to use ('openai', 'vertex', 'vllm', 'gemini', 'azure','aliyun')"
    )
    parser.add_argument("--model", type=str, help="Model to run topic generation with")

    parser.add_argument(
        "--seed_file", type=str, default="data/output/generation_1.md", help="Seed file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/generation_1.jsonl",
        help="Input data file",
    )
    parser.add_argument(
        "--prompt_file", type=str, default="prompt/generation_2.txt", help="Prompt file"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/generation_2.jsonl",
        help="Output result file",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_2.md",
        help="Output topics file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    generate_topic_lvl2(
        args.api,
        args.model,
        args.seed_file,
        args.data,
        args.prompt_file,
        args.out_file,
        args.topic_file,
        args.verbose,
    )
