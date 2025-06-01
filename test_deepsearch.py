import asyncio

from deepsearch.utils import extract_content
from deepsearch.graph import graph


async def solve(question):
    
    res = await graph.ainvoke({
        "task": question,
        "plan_string": None,
        "steps": [],
        "results": {},
        "result": None,
        "intermediate_result": None,
        "search_query": None,
        "needs_replan": False,
        "replan_iter": 0,
        "max_replan_iter": 1
    }, {"recursion_limit": 30})

    print(res)
    print(res["plan_string"])
    print(res["results"])

    response = res["result"]
    answer = extract_content(response, "answer")
    if answer is None:
        print("Failed without an answer!")  
    print(f"The answer is: {answer}")


query = "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name? "
asyncio.run(solve(query))
