# About
This project assesses LLMs (especially ChatGPT) as "AI portfolio managers" by ranking their portfolio construction abilities. As we can't simply perform a backtest because of temporal leakage, the only possible way is throught live observation. 
The system is built in order to work with several possible prompts, as it may influence it's behaviour. 
This method is suboptimal; agentic approaches are now known to yield better performance [see github.com/virattt/ai-hedge-fund].

# How to run
The "pipeline" script is responsable for calling the LLM and parsing the ranking. Then, a CI/CD with cron sends data to the UI [A more robust setup could use an orchestrator sending data to a NoSQL database].
