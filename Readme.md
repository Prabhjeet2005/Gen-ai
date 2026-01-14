2:28:00


# PRE-Requisites
## PRE-Requisites 1. Neural Network ("The Brain")
Mimics Human Brain, It has layer of Neurons(Hold a Number)
- Input_Layer | Hidden_Layer | Output_Layer
- *Input Layer :* Takes Data (Pixels of Image)
- *Hidden Layer :* Each Layer Finds Patterns (edges -> shapes -> faces)
- *Output Layer :* Final Decision (Eg: Number 9)

### Different Neural Networks:-
1. Images: CNNs [identify objects, faces, patterns in pictures]
1. Text[NLP]: RNNs & Transformers - *Recurrent Neural Network*  [understand, translate, generate Human Language]
1. Audio: [analyze speech, music, sounds]
### IMPORTANCE: 
Traditional ML fails with unstructured data(images/text)

### INTERVIEW ANSWER
- Deep Learning is subset of Machine Learning inspired by neural structure of human brain
- Traditional ML Models hit Performance Plateau early meaning more data don't yield significant results
- Deep Learning Model Improve with more the data provided because uncover more pattern
- **Why ML Performance Plateau?** 
1. Rely on Manual feature Extraction tell model what to look for, if miss a pattern 1 million more data won't help model see it
2. Deep Learning does Automated Feature Extraction, Finds own patterns, more data = more patterns


## PRE-Requisites 2. Old Way - RNNs & LSTMs [Read text linearly, Slow and Forget Early Words]
Before tranformers RNNs [Recurrent Neural Network] were used for text
- Process data word by word 
- they were "slow" and had "Short term memory"
- Can't remember context from starting of long paragraph
### Problem with text
- In Image -> All Pixels already present
- In Text -> Sentence -> Word Comes One By One

### How RNN Works?
- Short Term Memory
- Eg: The->Remember it, Read "Cat" combine with "The" to get "The Cat", Read "Sat" combine with "The Cat" to get "The Cat Sat"

### Major Flaw RNN
The Major Flaw (**The Vanishing Gradient**): By the time the RNN reads the 100th word in a paragraph, it has forgotten the 1st word. It couldn't write long essays or code.
#### Why this flaw Happens? 
[Just Like Chinese Whispers]
- Happens Due to Backpropogation (Network learns from its own errors)
- When the model tries to send the error signal backwards from the end of a long sequence to the beginning to update its weights, it effectively multiplies many small numbers (gradients < 1) together.
- Mathematically, this causes the signal to shrink exponentially until it becomes zero. As a result, the model fails to learn the relationship between the beginning and the end of the sequence, causing it to 'forget' early context."



## PRE-Requisites 3. Transformer [Reads text in parallel, Fast, Remember Everything]
- **CORE CONCEPT:** Instead of reading word by word, It reads entire sentence at once

### A. Attention Mechanism [Noisy party focus on friend voice filter out rest]
- Attention Mechanism focuses on every other word to figure out the context
- Eg: "I went to bank to deposit money"
- Money -> High Score, Deposit -> High Score, I -> Low Score
- **RESULT:** It Knows bank means financial Institution, not river side.

### B. Self Attention [Understanding Relationships]
- This lets model understand the grammar on its own without us teaching it the rules
- Eg: "Animal didn't cross the street because it was too tired."
- Model asks: What does "it" refer to?
- Self Attention connects "it" strongly to "animal" and weakly to "street"

### C. Architecture [Encoder-Decoder]
- Original Tranformers has 2 Parts
#### 1. ENCODER:
- Reads & Understands input (Eg: Listening)
- BERT(Bidirectional Encoder Representation from Transformers) uses only the Encoder (Great for understanding/classification)
#### 2. DECODER:
- Generates the output (Eg: Speaking)
- GPT(Generative Pre Trained Tranformer) uses only Decoder (Great for writing/generating)

## Pre-Requisites 4. Softmax & Temperature
### 1. Softmax (Output Layer) [Raw Number -> %]
- In Supervised ML Output is Hard Decision "Cat" or "Dog"
- In GenAi (LLMs), The Output is **Probability Distribution**
- Model Don't pick 1 word, it assigns % chance to every word in dictionary
- Eg: INPUT: "The sky is ..."
- MODEL OUTPUT: "blue"->80%, "cloudy"->10%, "pizza":0.01%

### 2. Temperature [How much Risk Model can take]
Parameter from 0 to 1 tells how risky the model acts
#### A. Temperature = 0 (Safe/Deterministic)
- Model always pick Highest Probability & give exact same answer every time
- USE CASE: Coding Assistant, Math Solvers (2+2 always equal 4)

#### B. Temperature = 1 (Creative/Random)
- Model Takes Risks
- It might pick "cloudy" or "pizza" just to be different
- USE CASE: Storytelling, poetry, Brainstorming Ideas

---
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
- TRADITIONAL LLM -> Focus only on text
- Now, MultiModel LLM -> Text, Images, Videos, Audios, Code

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
```File: 1 Data Preprocessing and Cleaning/text_preprocessing.ipynb```
- The complete roadmap of a project: 
- gathering data $\rightarrow$ cleaning it $\rightarrow$ choosing a model $\rightarrow$ building the app $\rightarrow$ deploying it to the web.
---
## Data representation & vectorization 
```File: 1 Data Preprocessing and Cleaning/data_representation_vectorization_for_model_training.ipynb```
- Converting text into numbers (vectors) because computers/models only understand math, not English.
- PROBLEM: data into MEANINGFUL numbers

1. **Feature Extraction from text/images**
2. **It's need -> GenAi is maths model needs numbers/vectors**
---
3. **Why difficult**
- **A. Text:** 
- Eg: "Hi I am Prabhjeet" Hi->0,I->1 and so on.. "Hello his name is ABC" here no correlation with previous string
- the model might think "I" (1) is bigger than "Hi" (0), which is math nonsense.
- Also, standard numbering loses context. "Bank" in "River Bank" and "Bank" in "HDFC Bank" would get the same number, confusing the model.
- **B. IMAGE:** 
- Suppose handwritten digit recognisation 0-255 color range of pixels, suppose 28x28 image size equals 784 so 784 neural network and then make the prediction
- A 28x28 image is a 2D grid. To feed it to a simple network, you have to "flatten" it into a long line of 784 numbers. In doing so, you often lose the spatial information (e.g., that pixel [0,0] is next to pixel [0,1]).
- **C. Audio:** 
- decibal V/s Frequency table create from (db v/s Hz graph) of audio 
- Audio is just a wave over time. To make it useful, we have to convert it into a "picture" of sound (Spectrogram - Decibels vs Frequency) so the model can "see" the sound patterns.
---
**4. Core Idea**
- The core idea is not just assignment (1, 2, 3), but ***Mapping***. We want to map every piece of data (word, image, or sound) to a point in a multi-dimensional graph (a coordinate system).
- The Rule: If two things are similar in meaning, their numbers (vectors) should be close together in math.
- ***Visualizing the Core Idea: Imagine a 3D graph:***
- X-axis: Is it living?
- Y-axis: Is it human?
- Z-axis: Is it royalty?

- King might be [0.9, 0.9, 0.9], Queen might be [0.9, 0.9, 0.8] (Very close!), Apple might be [0.1, 0.0, 0.0] (Far away)
---
### **5. Techniques**
#### **A. Text Techniques**
1. **One-Hot Encoding (Oldest):**
- Creates a massive list of zeros with a single '1'.
- Cons: Huge memory, no semantic meaning.

2. **Bag of Words (BoW):**
- Create Sparse Matrix like One Hot Encoding but Counts word frequency. "How many times did 'Prabhjeet' appear?"
- Can be used for sentiment analysis
- Cons: Ignores grammar ("Dog bit Man" = "Man bit Dog").
- N-grams: It is the parameter that tells N words as 1 token

3. **TF-IDF:**
- Counts frequency but lowers the score of useless words like "the" or "is".
- Cons: Still no semantic context.

4. **Word Embeddings (Word2Vec / GloVe):**
- Can Extract Semantic Information
- The breakthrough. Uses a small neural network to predict words. It learns that "King" and "Queen" are related.
- Deep Learning Approach
- **CBOW (Cont. Bag of Words) + SkipGram** 
##### **A. CBOW:** 
"The Fill-in-the-Blank Game" The model looks at the surrounding context words and tries to guess the missing target word.
- Input: Context words ("The", "quick", "fox", "jumps").
- Target: "brown".
- Logic: Find the most probable center word.
- Best For: Smaller datasets. It is faster to train.

##### **B. Skip-gram**
"The Reverse Game" The model looks at the word and tries to guess the surrounding context words.
- Input: Center word ("brown").
- Target: Context ("The", "quick", "fox", "jumps").
- Logic: This is harder. It forces the model to understand the word "brown" very deeply to know what words usually hang out near it.
- Best For: Large datasets. It is much better at understanding rare words.


5. **Transformers (BERT / GPT) - Current Standard:**
- *Contextual Embeddings:* It looks at the whole sentence at once. The vector for "Apple" changes depending on if you are talking about fruit or the iPhone company.

#### **B. Image Techniques**
- Pixel Values: Raw numbers (0-255).
- CNNs (Convolutional Neural Networks): Extracts features like "Edges," "Textures," and "Shapes" instead of just raw pixels.
- ViT (Vision Transformers): Splitting images into "patches" (like words in a sentence) and processing them using the same logic as GPT.

#### **C. Audio Techniques**
- Waveform: Raw amplitude numbers.
- Spectrograms / Mel-Spectrograms: Converting audio to an image (Heatmap of frequencies). This is what 99% of Audio AI models actually "look" at.




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



