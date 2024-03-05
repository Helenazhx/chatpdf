from langchain.prompts import ChatPromptTemplate


prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{info}

用户问：
{query}

请用中文回答用户问题。
"""

# def prompt_template_to_prompt(template):
#     return ChatPromptTemplate.from_template(template)

def build_prompt(template=prompt_template, **kwargs):
    """将 Prompt 模板赋值"""
    prompt_template = ChatPromptTemplate.from_template(template)
    input_val = dict()
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        input_val[k] = val
    return prompt_template.format_messages(info=input_val["info"], query=input_val["query"])
