import time
from tools import tools_map
from prompt import gen_prompt, user_prompt
from model_api import ModelAPI
from dotenv import load_dotenv


load_dotenv()
api = ModelAPI()


def parse_thought(response):
    try:
        thoughts = response.get("thoughts")
        plan = thoughts.get("plan")
        reasoning = thoughts.get("reasoning")
        criticism = thoughts.get("criticism")
        observation = response.get("observation")
        prompt = f"plan:{plan}\nreasoning:{reasoning}\ncriticism:{criticism}\nobservation:{observation}"
        return prompt
    except Exception as e:
        print("Parse thoughts error")
        return "".format(e)


def agent_execute(query, max_query_time=10):
    cur_query_time = 0
    chat_history = []
    agent_scratch = {
        "steps": [],
        "metrics": {},
        "warnings": []
    }
    while cur_query_time < max_query_time:
        cur_query_time += 1

        # 生成prompt
        prompt = gen_prompt(query, agent_scratch)
        start_time = time.time()
        print("*********{}Calling LLM......".format(cur_query_time), flush=True)
        response = api.chat(prompt, chat_history)
        end_time = time.time()
        print("{}Calling end......Cost: {} s".format(cur_query_time, round(end_time - start_time, 2)), flush=True)

        if not response or not isinstance(response, dict):
            print("Calling LLM error: ", response)
            continue


        # print(response)
        action_info = response.get("action")
        action_name = action_info.get("name")
        action_args = action_info.get("args")
        print("当前action_name:{}  action_args:{}".format(action_name, action_args))

        observation = response.get("observation")
        # 修改动作执行段
        try:
            if action_name == "finish":
                final_answer = action_args.get("answer")
                print(f"Final Answer: {final_answer}")
                return final_answer  # 提前返回终止循环
            
            func = tools_map.get(action_name)
            if not func:
                raise ValueError(f"未定义动作: {action_name}")
            
            call_function_result = func(**action_args)
        except Exception as e:
            call_function_result = f"动作执行失败: {str(e)}"

        execution_log = {
            "step": cur_query_time,
            "action": action_name,
            "args": action_args,
            "result": str(call_function_result),  # 截断避免过长
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        agent_scratch["steps"].append(execution_log)
        # print(agent_scratch)
        assistant_msg = parse_thought(response)
        chat_history.append([user_prompt, assistant_msg])
    if cur_query_time == max_query_time:
        print("本次任务失败")
    else:
        print("任务完成")


def main():
    max_query_time = 10
    while True:
        query = input("请输入：")
        if query == 'exit':
            return
        agent_execute(query, max_query_time)


if __name__ == '__main__':
    main()

