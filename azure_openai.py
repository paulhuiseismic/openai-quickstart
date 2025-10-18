import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Create Azure OpenAI client using environment variables
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# # 创建一个名为 "Math Tutor" 的助手，它是一个个人数学辅导老师。这个助手能够编写并运行代码来解答数学问题。
# assistant = client.beta.assistants.create(
#     name="Math Tutor",
#     instructions="You are a personal math tutor. Write and run code to answer math questions.",
#     tools=[{"type": "code_interpreter"}],  # 使用工具：代码解释器
#     model=os.getenv("AZURE_MODEL"),  # 使用环境变量中配置的模型
# )
