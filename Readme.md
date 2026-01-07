# 1. Introduction to Generative AI
## What is GenAi?
- GenAi is Ai that can create new content [text,images,code] rather than just analyzing existing data like classification



------------------------------------------------------------------------------
# 2. Data Preprocessing and Embeddings
## Data Preprocessing & Cleaning
- The complete roadmap of a project: 
- gathering data $\rightarrow$ cleaning it $\rightarrow$ choosing a model $\rightarrow$ building the app $\rightarrow$ deploying it to the web.

## Data representation & vectorization
- Converting text into numbers (vectors) because computers/models only understand math, not English.


------------------------------------------------------------------------------
# 3. Introduction to Large Language Models
- Deep Learning models trained on massive amounts of internet text that can understand and generate human-like language (e.g., GPT-4, Llama).
## Transformer-Attention: 
- The "brain"  modern AI. 
- "Attention" is model focus on the relevant words in a sentence (e.g., in "The bank of the river," knowing "bank" means land, not money).

## How ChatGPT is trained? 
- Pre-training (reading the internet) + Fine-tuning (learning to follow instructions) + RLHF (learning from human feedback).
------------------------------------------------------------------------------
# 4. Huggingface Platform and its API
- The "GitHub of AI." A platform where people host open-source models and datasets for free.

## Fine-tuning using pretrain models
- Taking a smart model (like BERT) and training it slightly more on your specific data to make it an expert in your domain.

## A. Project: Text summarization: 
- Building an app that takes a long paragraph and gives you a short summary.

## B. Project: Text to Image: 
- Using models (like Stable Diffusion) to generate pictures from a text description.


------------------------------------------------------------------------------
# 5. Complete Guide to Open AI

## A. Project: Telegram bot: 
- Connecting the AI to Telegram so you can chat with your bot on your phone.

## B. Project: Fine-tuning GPT-3: 
- Teaching GPT-3 specific examples so it behaves exactly how you want (e.g., talking like a pirate).


------------------------------------------------------------------------------
# 6. Mastering Prompt Engineering

- asking AI questions in a specific way to get the best possible answers.



------------------------------------------------------------------------------
# 7. Master Vector Database for LLMs

- ***Vector Databases***: Special databases to store "meaning" (vectors) rather than just keywords, allowing for semantic search.



------------------------------------------------------------------------------
# 8. LangChain - Basic to Advance

- ***LangChain***: A framework that acts as glue. It connects LLMs (brain) to other tools (calculators, Google search, PDFs) to build complex apps.

## Chains in LangChain: 
- Linking multiple steps together (e.g., Step 1: Translate to English -> Step 2: Summarize it).

## LangChain Agents: 
-  "tools" (like Google Search) decide which tool to use to answer a question.

## Memory in LangChain: 
- chatbot "short-term memory" so it remembers what you said 5 minutes ago in the conversation.

------------------------------------------------------------------------------
# 9. Learn to use Open Source LLMs




------------------------------------------------------------------------------
# 10. Retrieval Augmented Generation (RAG)

## Most important technique for devs
- It is fetching data from your PDF/Database and giving it to the AI to answer questions about your specific data.



------------------------------------------------------------------------------
# 11. Fine Tuning LLMs
## Parameter Efficient Fine-Tuning (PEFT/LoRA): 
- A smart way to fine-tune huge models on a small GPU (like a laptop or Colab) without needing a supercomputer.




------------------------------------------------------------------------------
# 12. LlamaIndex - Basic to Advance
- Framework similar to LangChain but specialized in handling data (indexing) for LLMs.



------------------------------------------------------------------------------
# 13. End-to-End Projects

## Project: Financial Stock Analysis: 
- Building an AI that reads stock news/reports and gives insights using LlamaIndex.

## Project: End to End Medical Chatbot: 
- A capstone project. Creating a bot that answers medical queries using a medical book PDF (using RAG + Pinecone + LangChain).


------------------------------------------------------------------------------
# 14. LLM Apps Deployment
- AWS Bedrock: Amazon's service that lets you access powerful models (like Claude or Titan) via API without managing servers.



------------------------------------------------------------------------------
# 15. LLMOps
- "DevOps for AI." Managing the lifecycle, monitoring, and deployment of large language models in production.



