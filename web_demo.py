#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion
from prompt_utils import build_prompt
from vectordb_utils import InMemoryVecDB
from pdf_utils import extract_text_from_pdf
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from text_utils import split_text


db_directory="./vec_db/demo"
embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")

def init_db(file):
    documents = extract_text_from_pdf(file.name)
    # documents = split_text(paragraphs, 500, 100)
    vec_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_directory
        )
    print("test:", vec_db)
    vec_db.persist()


def chat(user_input, chatbot, context, search_field):
    vec_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings
    )
    search_results = vec_db.similarity_search(user_input, 2)[0].page_content
    search_field = search_results
    prompt = build_prompt(info=search_results, query=user_input)
    response = get_completion(prompt, context)
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context, search_field


def reset_state():
    return [], [], "", ""


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">ChatPDF</h1>""")

        with gr.Row():
            with gr.Column():
                fileCtrl = gr.File(label="上传文件", file_types=[',pdf'])

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
            with gr.Column(scale=2):
                # gr.HTML("""<h4>检索结果</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="检索结果...", lines=10)
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=2)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")

        context = gr.State([])
        fileCtrl.upload(init_db, inputs=[fileCtrl])

        submitBtn.click(chat, [user_input, chatbot, context, search_field],
                        [user_input, chatbot, context, search_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

    demo.queue().launch(share=True, server_name='0.0.0.0', server_port=4321, inbrowser=True)


if __name__ == "__main__":
    main()
