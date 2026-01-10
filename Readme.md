# 1. Introduction to Generative AI
## What is GenAi?
- GenAi is Ai that can create new content [text,images,code] rather than just analyzing existing data like classification
---
**Structured V/S Unstructured Data**
- Structured Data is Row and Cols like: SQL DB, CSV Files
- Unstructured Data is full of messy data like: Email, Audio, PDF, Images which LLM's can understand.
---
**GENAI V/S LLM**
- GENAI: Broad Term For New Content Generation
- LLM: Specific Type of GenAI focusing on Language/Text.

---
**WHY GEN-AI model required?**
- Understand Complex Pattern
- Content Generation
- Build Powerful Apps 

---
**Discriminative V/s Generative Model**
1. **DISCRIMINATIVE MODEL**
- It Classifies
- Care More about boundaries like Classify Dog or Cat
- Line B/w classes

2. **GENERATIVE MODEL**
- It Generates or creates
- Care About Distribution (Shape) of data
- How Data Generated, uncover underlying patterns

---
### End To End GenAi Pipeline
```
Data Cleaning -> Model Selection -> Domain Adapt [Prompt/RAG/Fine Tuning] -> Application Integration -> LLMOps
```

**1. Data Collection & Cleaning**
- What you do: Collect data (PDFs, text files, databases).
- Cleaning: Remove HTML tags, emojis, and useless text.
- Chunking: Break large text into smaller pieces because LLMs have a limit on how much they can read at once.

**2. Model Selection (The Brain)**
- Closed Source: OpenAI (GPT-4), Gemini. (Easy, powerful, but paid).
- Open Source: Llama 3, Mistral. (Free, runs on your hardware, guarantees privacy).

**3. Domain Adaptation [Prompt/RAG/Fine-Tuning]**
- This is where you make the model an "expert" in your specific field.
- Prompt Engineering: Writing smart instructions ("You are a medical assistant...").
- RAG (Retrieval Augmented Generation): Connecting the model to your live data (PDFs/Database) so it answers based on facts, not hallucinations. This is what you will use most as a developer.
- Fine-Tuning: Retraining the model slightly to learn a new language or style.

**4. Application Integration (The Code)**
- Use frameworks like LangChain or LlamaIndex to connect the AI model to your Node.js/Express backend.
- You build the UI (React) where users type questions.

**5. Deployment & Ops (LLMOps)**
Moving from "it works on my laptop" to "it works for 10,000 users."
- Deploy: Hosting the model on AWS/Google Cloud.
- Monitoring: Watching if the AI starts talking crazy or if it's too slow.

---
**DATA COLLECTION**
- If Data Is Less, Techniques To Generate More Data :-
1. Data Augmentation
- If Less Data Use Data Augmentation [Replace with Synonyms]. 
- Eg: Hello, I am Prabhjeet ---> Hi, I am Prabhjeet
2. Biagram Flip
- Eg: I am Prabh --> Prabh is my name
3. Back translate [1 Language to Another Then Back to same language again]
4. Additional Data/ Noise
---
**Data Preprocessing**
1. Cleanup: Html, Emoji, Spelling Correction
2. Basic Preprocessing: Tokenization [Sentence, World Level Tokenization]
3. Optional PreProcessing: Stop Word Removal, Punctuation, Language Detection, Stemming[Less Used], Lemmatization[More Used]
4. Advanced Preprocessing: 
- Parts of Speech Tagging [Grammatical Label Each word to understand Context eg: Noun, Verb, Adjective], 
- Parsing [Resolves Ambiguity, Finds Dependency] 
- Coreference Resolution [Figure out he,she,it,they refer to in text. Eg: Dog Ate bone because it was hungry. (it->dog)]

- NOTE: 
1. Stemming:[Chop Suffix/prefix play,playing,plays -> play], Sometimes Word Not make sense.
2. Lemmatization: [Context aware, running,ran,runs -> run ]

---
**Feature Engineering**
1. Text Vectorization: [Text -> Vector(Numbers) Convert]
- **One Hot Encoding**: No Context Understand. Every Word Different Vector, Make List of all words [Apple,Banana,Dog] Apple->[1,0,0], Banana -> [0,1,0], Dog -> [0,0,1]
- **Bag Of Words**: No Context Count Occurence of Words. Eg: The dog bit the man {the:2,dog:1,bit:1,man:1}
- **TFIDF [Term Frequency - Inverse Document Frequency]** : Similar to Bag of Words and assumes Rare Words are More important. Eg: "the","is" are given Low score and words like "algorithm", "genai" are given high score.
- **Word2Vec:** Use Neural Network to learn relationships. Capture Semantic Meaning. Words With Similar Meaning Together. Eg: King - Man + Woman = Queen [Take King vector remove man and add woman then we get queen vector]
- **Transformer Model**: Understand Context Perfectly, Transformers create Dynamic Vectors based on context. Eg: "River Bank" [Here Bank is Nature/Water related],"Axis Bank" [Here Bank is Financial/money related]
---

**Modelling**
- Open Source
- Closed Source

---
**Evaluation**
1. Intrinsic: [Lab test]
- Perplexity: How confused Model is. Lower the Better
- BLEU/ROUGE Score: Compare Ai text to "perfect" human written answer & check Overlap
2. Extrinsic: [After Deployment, In Production]
- human Feedback
- A/B Testing: Show Model A to 50% of People and Model B to other 50% of users and check which group more happy.

--- 
**Deployment**
- Monitoring: Keep it updated with World [Data Drift]
- Retraining: Retrain Bad Interactions
---

---
### COMMON TERMS
1. Corpus: Entire Text
2. Vocabulory: Unique Word [If Repeat Word Take First Occurence]
3. Documents: Each Line is new Document
4. Word: Single Word
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



