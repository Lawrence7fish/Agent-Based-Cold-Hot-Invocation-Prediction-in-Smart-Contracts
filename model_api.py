import json

from openai import OpenAI
import dotenv
import os
from dashscope.api_entities.dashscope_response import Message
from prompt import user_prompt


class ModelAPI(object):
    def __init__(self):
        dotenv.load_dotenv()
        self.api_key = os.environ['API_KEY']
        self._model_name = os.environ['MODEL_NAME']
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=os.getenv("API_URL"),

        )

        self.max_retry_times = 5

    def chat(self, prompt, history):
        cur_retry_time = 0
        while cur_retry_time < self.max_retry_times:
            cur_retry_time += 1
            try:
                # 采用Message类把信息组装好
                messages = [Message(role='system', content=prompt)]
                for his in history:
                    messages.append(Message(role='user', content=his[0]))
                    messages.append(Message(role='assistant', content=his[1]))
                messages.append(Message(role='user', content=user_prompt))
                completion = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=0.3  # 降低随机性
                )
                """
                {
                    "id":"chatcmpl-8d87fffd-a5bc-9dc9-90fe-6e6f79bbb24f",
                    "choices":[
                        {
                            "finish_reason":"stop",
                            "index":0,
                            "logprobs":null,
                            "message":{
                                "content":"{\n
                                    \"actions\": {\n
                                            \"name\": \"read_file\",\n        
                                            \"args_name\": \"travel_planner_template\"\n    
                                    },\n    \
                                    "thought\": {\n        
                                        \"plan\": \"根据提供的模板，生成适合从广州到日本的7天旅游计划。\",\n        
                                        \"criticism\": \"在开始之前，我需要确定是否有一个可用的模板或指南来指导如何制定这样的旅行计划。如果没有，我将不得不从头开始研究并创建一个有效的计划。\",\n        
                                        \"speak\": \"正在查找关于从广州到日本的旅游计划的模板或指南。\",\n        
                                        \"reasoning\": \"首先，我们需要找到一个合适的模板或者指南来制定这个旅游计划，以便我们能够为用户提供一个可行且详细的行程安排。\"\n   
                                         }\n}",
                            "refusal":null,
                            "role":"assistant",
                            "audio":null,
                            "function_call":null,
                            "tool_calls":null}}],
                    "created":1741333607,
                    "model":"qwen2.5-1.5b-instruct",
                    "object":"chat.completion",
                    "service_tier":null,
                    "system_fingerprint":null,
                    "usage":{"completion_tokens":154,"prompt_tokens":540,"total_tokens":694,"completion_tokens_details":null,"prompt_tokens_details":null}}
                """
                result = completion.model_dump()['choices'][0]['message']['content']
                result = json.loads(result)
                return result
            except Exception as e:
                print("Calling LLM error:", e)

