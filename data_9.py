# -*- coding: utf-8 -*-
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 设置模型名称
model_name = "Maykeye/TinyLLama-v0"

# 使用认证令牌加载模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# 启用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 生成回答函数
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# /start 命令处理器
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Hi! I am a TinyLlama-powered bot. Ask me anything, for example: 'Tell me about cats.'"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)

# 用户消息处理器
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        # 调用 TinyLlama 生成回答
        response = generate_response(user_message)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    except Exception as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {e}")

# 初始化 Telegram Bot
TOKEN = "your token"  # 替换为你的 Telegram Bot Token
application = ApplicationBuilder().token(TOKEN).build()

# 添加处理器
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# 运行 Bot
application.run_polling()
