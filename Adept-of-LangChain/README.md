## Langchain-Core
### Шаблоны промптов
* `PromptTemplate`: `template`, `input_variables`
* `ChatPrompttTemplate`: `.from_template` или `prompt.format_messages()`
* `FewShotTPromptTemplate`: `examples = List`, `example - template`
### Агенты
* `from langchain.agents import Tool, initialize_agent, load_tools`
* `tool = Tool (name, func, description)`
* `agent = initialize_agent(tools, llm, verbose, max_iterations`
### Structured Output
* `from langchain.output_parsers import ResponseSchema, StructuredOutputParser`
* `schema = ResponseSchema (name, description)`
* `response_schemas = List`
* `output_parser = StructuredOutputParser.from_response_schemas(response_schemas)`
* `format_instructions = output_parser.get_format_instructions()`
