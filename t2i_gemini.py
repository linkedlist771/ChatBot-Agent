from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import re


load_dotenv()


# T2I_BASEURL
# T2I_API_KEY
T2I_BASEURL = os.getenv("T2I_BASEURL")
T2I_API_KEY = os.getenv("T2I_API_KEY")  


async def t2i_gemini(prompt: str) -> str:
    """
    调用 Gemini 生成图片
    
    Args:
        prompt: 图片描述提示词
        
    Returns:
        图片 URL
    """
    client = AsyncOpenAI(base_url=T2I_BASEURL, api_key=T2I_API_KEY)
    payload = {
        "model": "gemini-3-pro-imagen",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = await client.chat.completions.create(**payload)
    content = response.choices[0].message.content
    
    # 解析图片 URL: ![image](https://xxx.jpg)
    match = re.search(r'!\[image\]\((https?://[^\)]+)\)', content)
    if not match:
        raise ValueError(f"No image URL found in response: {content}")
    
    return match.group(1)


async def example_usage():
    image_url = await t2i_gemini("画一头猪")
    print(f"图片 URL: {image_url}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
