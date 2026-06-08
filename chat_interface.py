"""
AI对话Web界面 - 用于测试微调后的本地模型
基于Gradio构建，支持切换原始模型与手术后模型进行对比测试
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Generator, List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 尝试导入gradio，如未安装则给出提示
try:
    import gradio as gr
except ImportError:
    print("错误：未安装gradio。请运行: pip install gradio")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent

# 默认模型路径
DEFAULT_ORIGINAL_MODEL = SCRIPT_DIR / "Qwen3-1.7B"
DEFAULT_SURGERY_MODEL = SCRIPT_DIR / "surgery-qwen3-1.7b"


class ModelManager:
    """管理模型加载与对话生成"""

    def __init__(self):
        self.models: Dict[str, tuple] = {}  # name -> (model, tokenizer, device)
        self.current_model_name: Optional[str] = None

    def pick_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_model(self, name: str, model_path: Path, dtype: torch.dtype = torch.float16) -> str:
        """加载模型，返回状态信息"""
        if name in self.models:
            return f"模型 '{name}' 已加载"

        if not model_path.exists():
            return f"错误：模型路径不存在 {model_path}"

        try:
            device = self.pick_device()
            print(f"[{name}] 正在加载模型: {model_path}")
            print(f"[{name}] 使用设备: {device}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )

            # 检查是否有safetensors文件来估计模型大小
            safetensors_files = list(model_path.glob("*.safetensors"))
            if safetensors_files:
                model_size_gb = sum(f.stat().st_size for f in safetensors_files) / (1024**3)
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

                if torch.cuda.is_available() and model_size_gb + 2 > gpu_mem_gb:
                    print(f"[{name}] 模型 ({model_size_gb:.1f}GB) 超过GPU显存 ({gpu_mem_gb:.1f}GB)，使用CPU加载")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        local_files_only=True,
                        trust_remote_code=True,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        local_files_only=True,
                        trust_remote_code=True,
                        device_map="auto",
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                    trust_remote_code=True,
                ).to(device)

            model.eval()
            self.models[name] = (model, tokenizer, device)
            return f"✅ 模型 '{name}' 加载成功 ({device})"

        except Exception as e:
            return f"❌ 加载模型 '{name}' 失败: {str(e)}"

    def unload_model(self, name: str) -> str:
        """卸载模型释放内存"""
        if name not in self.models:
            return f"模型 '{name}' 未加载"

        del self.models[name]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return f"✅ 模型 '{name}' 已卸载"

    def generate_stream(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> Generator[str, None, None]:
        """流式生成回复"""
        if model_name not in self.models:
            yield f"错误：模型 '{model_name}' 未加载"
            return

        model, tokenizer, device = self.models[model_name]

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(device)

            # 使用generate的streamer实现流式输出
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for text in streamer:
                generated_text += text
                yield generated_text

            thread.join()

        except Exception as e:
            yield f"生成错误: {str(e)}"

    def generate_once(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> str:
        """非流式生成回复"""
        if model_name not in self.models:
            return f"错误：模型 '{model_name}' 未加载"

        model, tokenizer, device = self.models[model_name]

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(device)

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output[0, inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        except Exception as e:
            return f"生成错误: {str(e)}"


# 全局模型管理器
model_manager = ModelManager()


def create_ui() -> gr.Blocks:
    """创建Gradio界面"""

    with gr.Blocks(
        title="模型脑手术 - AI对话测试界面",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        css="""
        .model-status { font-size: 0.9em; padding: 8px; border-radius: 6px; }
        .model-loaded { background: #dcfce7; color: #166534; }
        .model-error { background: #fee2e2; color: #991b1b; }
        .chat-container { min-height: 500px; }
        .header-text { text-align: center; margin-bottom: 1rem; }
        .compare-row { gap: 1rem; }
        """
    ) as demo:

        gr.Markdown("""
        # 🧠 模型脑手术 - AI对话测试界面

        用于测试和对比原始模型与手术后（微调后）模型的对话效果。
        """)

        with gr.Tab("单模型对话"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型配置")

                    model_name_input = gr.Textbox(
                        label="模型名称",
                        value="手术后模型",
                        placeholder="给模型起个名字",
                    )

                    model_path_input = gr.Textbox(
                        label="模型路径",
                        value=str(DEFAULT_SURGERY_MODEL),
                        placeholder="模型文件夹路径",
                    )

                    with gr.Row():
                        load_btn = gr.Button("🚀 加载模型", variant="primary")
                        unload_btn = gr.Button("🛑 卸载模型", variant="secondary")

                    load_status = gr.Textbox(
                        label="加载状态",
                        value="未加载",
                        interactive=False,
                        elem_classes=["model-status"],
                    )

                    gr.Markdown("### 生成参数")
                    max_tokens = gr.Slider(
                        label="最大生成token数",
                        minimum=32,
                        maximum=2048,
                        value=512,
                        step=32,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                    )
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                    )
                    stream_output = gr.Checkbox(
                        label="流式输出",
                        value=True,
                    )

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=600,
                        elem_classes=["chat-container"],
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="输入消息",
                            placeholder="输入你的问题...",
                            scale=8,
                            lines=2,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1, size="lg")

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空对话")
                        system_msg = gr.Textbox(
                            label="系统提示词（可选）",
                            placeholder="输入系统提示词...",
                            scale=2,
                        )

            # 状态存储
            state_messages = gr.State([])

            def on_load_model(name: str, path: str) -> str:
                return model_manager.load_model(name, Path(path))

            def on_unload_model(name: str) -> str:
                return model_manager.unload_model(name)

            def on_send_message(
                message: str,
                messages: List[Dict],
                model_name: str,
                max_tok: int,
                temp: float,
                tp: float,
                stream: bool,
                system: str,
            ):
                if not message.strip():
                    return messages, messages, ""

                # 添加系统消息（如果有）
                current_messages = messages.copy()
                if system and system.strip() and not current_messages:
                    current_messages.append({"role": "system", "content": system})

                current_messages.append({"role": "user", "content": message})

                if stream:
                    # 流式输出 - 先添加空的助手消息
                    current_messages.append({"role": "assistant", "content": ""})
                    yield current_messages, current_messages, ""

                    full_response = ""
                    for partial in model_manager.generate_stream(
                        model_name, current_messages[:-1], max_tok, temp, tp
                    ):
                        full_response = partial
                        current_messages[-1]["content"] = full_response
                        yield current_messages, current_messages, ""
                else:
                    response = model_manager.generate_once(
                        model_name, current_messages, max_tok, temp, tp
                    )
                    current_messages.append({"role": "assistant", "content": response})
                    yield current_messages, current_messages, ""

            def on_clear():
                return [], [], ""

            load_btn.click(
                on_load_model,
                inputs=[model_name_input, model_path_input],
                outputs=load_status,
            )

            unload_btn.click(
                on_unload_model,
                inputs=model_name_input,
                outputs=load_status,
            )

            send_btn.click(
                on_send_message,
                inputs=[
                    msg_input, state_messages, model_name_input,
                    max_tokens, temperature, top_p, stream_output, system_msg,
                ],
                outputs=[chatbot, state_messages, msg_input],
            )

            msg_input.submit(
                on_send_message,
                inputs=[
                    msg_input, state_messages, model_name_input,
                    max_tokens, temperature, top_p, stream_output, system_msg,
                ],
                outputs=[chatbot, state_messages, msg_input],
            )

            clear_btn.click(on_clear, outputs=[chatbot, state_messages, msg_input])

        with gr.Tab("双模型对比"):
            gr.Markdown("""
            ### 并排对比两个模型的回答
            同时加载两个模型，输入一个问题，对比它们的回答差异。
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 模型A（左）")
                    model_a_name = gr.Textbox(label="名称", value="原始模型")
                    model_a_path = gr.Textbox(label="路径", value=str(DEFAULT_ORIGINAL_MODEL))
                    load_a_btn = gr.Button("加载模型A", variant="primary")
                    status_a = gr.Textbox(label="状态", value="未加载", interactive=False)

                with gr.Column():
                    gr.Markdown("#### 模型B（右）")
                    model_b_name = gr.Textbox(label="名称", value="手术后模型")
                    model_b_path = gr.Textbox(label="路径", value=str(DEFAULT_SURGERY_MODEL))
                    load_b_btn = gr.Button("加载模型B", variant="primary")
                    status_b = gr.Textbox(label="状态", value="未加载", interactive=False)

            with gr.Row():
                compare_prompt = gr.Textbox(
                    label="测试提示词",
                    placeholder="输入要对比测试的提示词...",
                    lines=3,
                    scale=3,
                )
                compare_btn = gr.Button("🔄 开始对比", variant="primary", scale=1, size="lg")

            with gr.Row(elem_classes=["compare-row"]):
                with gr.Column():
                    gr.Markdown("**模型A 输出**")
                    output_a = gr.Textbox(
                        label="",
                        lines=20,
                        interactive=False,
                    )

                with gr.Column():
                    gr.Markdown("**模型B 输出**")
                    output_b = gr.Textbox(
                        label="",
                        lines=20,
                        interactive=False,
                    )

            with gr.Row():
                compare_max_tokens = gr.Slider(
                    label="最大生成token数", minimum=32, maximum=1024, value=256, step=32
                )
                compare_temp = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.7, step=0.1
                )

            def on_load_a(name: str, path: str) -> str:
                return model_manager.load_model(name, Path(path))

            def on_load_b(name: str, path: str) -> str:
                return model_manager.load_model(name, Path(path))

            def on_compare(
                prompt: str,
                name_a: str,
                name_b: str,
                max_tok: int,
                temp: float,
            ):
                if not prompt.strip():
                    return "请输入提示词", "请输入提示词"

                messages = [{"role": "user", "content": prompt}]

                result_a = model_manager.generate_once(name_a, messages, max_tok, temp)
                result_b = model_manager.generate_once(name_b, messages, max_tok, temp)

                return result_a, result_b

            load_a_btn.click(on_load_a, inputs=[model_a_name, model_a_path], outputs=status_a)
            load_b_btn.click(on_load_b, inputs=[model_b_name, model_b_path], outputs=status_b)
            compare_btn.click(
                on_compare,
                inputs=[compare_prompt, model_a_name, model_b_name, compare_max_tokens, compare_temp],
                outputs=[output_a, output_b],
            )

        with gr.Tab("批量测试"):
            gr.Markdown("""
            ### 批量测试提示词列表
            从文件中加载提示词列表，批量测试模型回答。
            """)

            with gr.Row():
                batch_model_name = gr.Textbox(label="模型名称", value="手术后模型")
                batch_prompts_file = gr.File(
                    label="提示词文件（每行一个）",
                    file_types=[".txt"],
                )

            with gr.Row():
                batch_max_tokens = gr.Slider(
                    label="最大生成token数", minimum=32, maximum=512, value=128, step=32
                )
                batch_temp = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.7, step=0.1
                )

            run_batch_btn = gr.Button("▶️ 运行批量测试", variant="primary")

            batch_output = gr.Dataframe(
                headers=["提示词", "模型回答"],
                label="测试结果",
                wrap=True,
            )

            def on_batch_test(
                model_name: str,
                prompts_file,
                max_tok: int,
                temp: float,
            ):
                if model_name not in model_manager.models:
                    return [["错误", f"模型 '{model_name}' 未加载"]]

                if prompts_file is None:
                    return [["错误", "请上传提示词文件"]]

                try:
                    with open(prompts_file.name, "r", encoding="utf-8") as f:
                        prompts = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    return [["错误", f"读取文件失败: {str(e)}"]]

                results = []
                for prompt in prompts:
                    messages = [{"role": "user", "content": prompt}]
                    response = model_manager.generate_once(model_name, messages, max_tok, temp)
                    results.append([prompt, response])

                return results

            run_batch_btn.click(
                on_batch_test,
                inputs=[batch_model_name, batch_prompts_file, batch_max_tokens, batch_temp],
                outputs=batch_output,
            )

        with gr.Tab("预设测试"):
            gr.Markdown("""
            ### 使用内置的harmful/harmless提示词进行测试
            快速验证模型手术效果。
            """)

            preset_model = gr.Textbox(label="模型名称", value="手术后模型")
            preset_type = gr.Radio(
                label="提示词类型",
                choices=["harmful（有害）", "harmless（无害）"],
                value="harmful（有害）",
            )
            preset_count = gr.Slider(
                label="测试数量", minimum=1, maximum=10, value=3, step=1
            )

            run_preset_btn = gr.Button("▶️ 运行预设测试", variant="primary")
            preset_output = gr.Dataframe(
                headers=["提示词", "模型回答"],
                label="测试结果",
                wrap=True,
            )

            def on_preset_test(
                model_name: str,
                preset: str,
                count: int,
            ):
                if model_name not in model_manager.models:
                    return [["错误", f"模型 '{model_name}' 未加载"]]

                # 导入提示词列表
                from harmful_prompts import harmful_prompts
                from harmless_prompts import harmless_prompts

                prompts = harmful_prompts if "harmful" in preset else harmless_prompts
                prompts = prompts[:count]

                results = []
                for prompt in prompts:
                    messages = [{"role": "user", "content": prompt}]
                    response = model_manager.generate_once(model_name, messages, 256, 0.7)
                    results.append([prompt, response])

                return results

            run_preset_btn.click(
                on_preset_test,
                inputs=[preset_model, preset_type, preset_count],
                outputs=preset_output,
            )

        gr.Markdown("""
        ---
        **使用提示**：
        - 首次使用请先加载模型
        - 模型加载需要一定时间，请耐心等待
        - 对比模式可同时加载两个模型进行A/B测试
        - 预设测试使用项目内置的harmful/harmless提示词
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="AI对话Web测试界面")
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="创建公开分享链接")
    parser.add_argument(
        "--auto-load",
        nargs="?",
        const=str(DEFAULT_SURGERY_MODEL),
        help="启动时自动加载模型（可指定路径，默认手术后模型）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 模型脑手术 - AI对话测试界面")
    print("=" * 60)

    demo = create_ui()

    # 自动加载模型
    if args.auto_load:
        model_path = Path(args.auto_load)
        model_name = "自动加载模型"
        print(f"\n正在自动加载模型: {model_path}")
        result = model_manager.load_model(model_name, model_path)
        print(result)

    print(f"\n启动Web界面: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
