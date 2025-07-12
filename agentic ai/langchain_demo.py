from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.llms.base import LLM
from euri_llm import generate_completion

# going to create a simple agent which can write a poem/ story - writing tool

@tool # decorator to tell langchain that this is a tool
def expert_writer(input):
    """this is my expert writer tool"""
    message = [{"role": "user", "content": f"Write a sort poem on {input}"}]
    return generate_completion(messages=message)








# going to create a simple agent which can do mathematical calculations - math tool

@tool # decorator to tell langchain that this is a tool
def expert_math(input):
    """this is my expert math tool"""
    result = eval(input,{"__builtins__":{}}, {})
    return result


tools = [expert_writer, expert_math]


class EuriaiLLM(LLM):
    def _call(self, prompt, stop=None):
        return generate_completion([{"role": "user", "content": prompt}])
    
    @property
    def _llm_type(self):
        return "euri-llm"


# here prompt template from langchain is used to create a prompt for the agent
prompt = PromptTemplate.from_template(
    """You are an intelligent agent with access to tools: {tool_names}

{tools}

Use this format strictly:
Thought: describe what you want to do
Action: the tool to use, one of [{tool_names}]
Action Input: the input in JSON format, e.g., {{"input": "2+2"}}
Observation: result of the action
... (repeat Thought/Action/Observation if needed)
When you have enough information, output ONLY:
Thought: I now know the final answer
Final Answer: the final response to the user

Begin!

Question: {input}
{agent_scratchpad}"""
)


agent = create_react_agent(
    llm = EuriaiLLM(),
    tools = tools,
    prompt = prompt,
    output_parser = ReActSingleInputOutputParser()
     #ReActSingleInputOutputParser is a class that parses the output of the agent
)

executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors = True,
    max_iterations = 5
)

response = executor.invoke({"input": "give me a poem based on earth and try to execute for 4*6 lines"})
print(response['output'])


"""
- here we are using the react agent to decide which operation to perform we are not manually telling the agent what to do
- here it will make the autonomous decision based on the input and the tools available to it

"""
