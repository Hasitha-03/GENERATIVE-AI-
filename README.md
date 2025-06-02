**Generative AI and LLMs: Architecture**

**Generative AI?**
Definition: Deep learning models that generate high-quality, new content (text, images, audio, 3D objects, music) based on patterns learned from training data.
Analogy: Like an artist who studies paintings to understand patterns and then creates original art.
Core Idea: Understands patterns and structures in existing data to produce new, relevant data

**Types of Generative AI Models & Examples**
Text Generation: Understand context and relationships between words/phrases.
Applications: Story generation, translation, summarization.
Example: GPT (Generative Pre-trained Transformer).
Image Generation: Text-to-Image: Generates images from text prompts (e.g., "a robot playing a piano").
Example: DALL-E.
Image-to-Image/Seed-based: Generates variations from seed images or random inputs.
Applications: realistic images from sketches, deepfakes.
Examples: GAN (Generative Adversarial Network), Diffusion Models.
Audio Generation: Generates natural-sounding speech, text-to-speech synthesis.
Example: WaveNet.
Other: 3D objects, music.


**Common Generative AI Architectures**
**RNNs (Recurrent Neural Networks):** Use sequential/time-series data. Loop-based design to remember previous inputs.
Applications: NLP, language translation, speech recognition. Fine-tuning: Adjust weights and structure.
**Transformers:** Deep learning models for text/speech translation in near real-time. Self-Attention Mechanism: Focuses on important parts of input, enables parallelization.
Fine-tuning: Typically involves training only final output layers. Example: GPT (a generative model using Transformer architecture).
**GANs (Generative Adversarial Networks):** Two submodels: 1. Generator: Creates fake samples. 2. Discriminator: Tries to distinguish real from fake samples.
Adversarial process improves both models. Useful for: Image and video generation.
**VAEs (Variational Autoencoders):** Encoder-decoder framework.
Encoder: Compresses input to a simplified latent space.
Decoder: Recreates original data from latent space.
Focuses on learning underlying data patterns; represents data using probability distributions.
Useful for: Art, creative design.
**Diffusion Models:** Probabilistic generative models.
Trained to generate images by learning to remove noise or reconstruct distorted examples.
Can generate highly creative images from noisy/low-quality inputs.

**Training Approaches Summary:**
RNNs: Loop-based design.
Transformers: Self-attention mechanism.
GANs: Competitive training (generator vs. discriminator).
VAEs: Characteristics-based (encoder-decoder).
Diffusion Models: Statistical properties, noise removal.
Reinforcement Learning (RL): Generative AI models use RL techniques during training to fine-tune and optimize performance

** Applications of Generative AI**
General:
Content Creation: Articles, blogs, marketing materials, visuals, videos.
Summarization: Condensing long documents.
Language Translation: More natural-sounding translations.
Chatbots & Virtual Assistants: More human-like and effective customer support.
Data Analysis (NLP): Uncover insights, suggest creative solutions.
Industry-Specific:
Healthcare: Analyze medical images, create patient reports.
Finance: Predictions and forecasts from financial data.
Gaming: Interactive elements, dynamic storylines.
IT: Create artificial data for training models.
Future: Personalized recommendations, drug discovery, smart homes, autonomous vehicles. 

**Large Language Models (LLMs):**
Definition: Foundation models using AI and deep learning with vast datasets (petabytes) and billions of parameters.
Capabilities: Generate text, translate, create content with minimal task-specific training.
Examples: GPT series, BERT, BART, T5 (Text-to-Text Transfer Transformer).
Architecture Basis: Most LLMs are based on the Transformer architecture.
GPT vs. ChatGPT:
GPT: Broader text generation; primarily supervised learning.
ChatGPT: Focused on conversational generation; uses supervised learning + RLHF (Reinforcement Learning from Human Feedback).
Caution: Can generate plausible but inaccurate information (hallucinations) and may reflect biases from training data.


**Tokenization & Data Loading**
1. Tokenization
Definition: The process of breaking down text (a sentence or document) into smaller pieces called tokens (words, characters, or subwords).
Tokenizer: The program/tool that performs tokenization (e.g., NLTK, spaCy).
Purpose: Helps models understand text better by converting it into a format suitable for processing

**Tokenization Methods**
1. Word-based Tokenization:
Text is split into individual words.
Advantage: Preserves semantic meaning.
Disadvantage: Large vocabulary size (e.g., "unicorn" and "unicorns" are different tokens), out-of-vocabulary (OOV) issues.
Examples: NLTK, spaCy tokenizers.
2. Character-based Tokenization:
Text is split into individual characters.
Advantage: Small vocabulary, handles OOV words.
Disadvantage: Single characters may not convey full meaning, increases input dimensionality and computational needs.
3. Subword-based Tokenization:
Combines advantages of word and character-based.
Frequently used words remain unsplit; infrequent words are broken into meaningful subwords.

Algorithms:
WordPiece: Evaluates benefits of splitting/merging symbols (used by BERT). ## prefix indicates attachment to the previous word.
Unigram: Starts with many possibilities, narrows down based on frequency.
SentencePiece: Segments text and assigns unique IDs (used by XLNet). _ prefix indicates a new word preceded by a space.

**Tokenization and Indexing in PyTorch (torchtext)**
Process:
Use get_tokenizer to tokenize text.
Use build_vocab_from_iterator to create a vocabulary from tokens.
Each unique token in the vocabulary gets a unique integer index.
vocab.get_stoi(): Returns a dictionary mapping words to their numerical indices.
vocab.set_default_index(vocab['<UNK>']): Sets a default index for unknown (OOV) words.
yield_tokens function: Often used to process data iterators and yield tokenized output.


Special Tokens & Padding
Special Tokens:
<UNK>: Unknown token.
<BOS> or <s>: Beginning of a sentence.
<EOS> or </s>: End of a sentence.
<PAD>: Padding token.
Padding:
Process of adding <PAD> tokens to sequences in a batch to make them all the same length (required by many models).
PyTorch: pad_sequence function.
padding_value: Value to use for padding (often index of <PAD>).
batch_first=True: Ensures batch dimension is the first dimension of the output tensor (e.g., [batch_size, sequence_length]). If False (default), it's [sequence_length, batch_size].


Data Loaders (PyTorch)
Purpose: Efficiently load, batch, shuffle, and preprocess data for model training.
Dataset (torch.utils.data.Dataset):
Starting point; represents a collection of data samples and labels.
Custom Dataset Class: Inherits from Dataset and implements:
__init__(self, data): Initialize with data.
__len__(self): Return total number of samples.
__getitem__(self, idx): Retrieve a sample at a given index.
DataLoader (torch.utils.data.DataLoader):
An iterator object for loading, shuffling, and batching data from a Dataset.
Key Parameters:
dataset: The custom Dataset object.
batch_size: Number of samples per batch.
shuffle=True: Randomly shuffles data before each epoch (good for training).
collate_fn: A function to customize how samples are batched (e.g., tokenization, padding, tensor conversion within the batch).
Collate Function (collate_fn):
Processes a list of samples (a batch) from the Dataset.
Common tasks: Tokenizing, numericalizing (converting tokens to indices), padding sequences to the same length, converting to tensors.
Allows data transformations to be done efficiently at batch level.


**Data Quality **
Data Quality: Accuracy, consistency, completeness.
Noise Reduction: Remove irrelevant/repetitive data, typos, tags.
Consistency Checks: Uniform usage for entities, terms.
Labeling Quality: Accurate labels for supervised tasks; clear guidelines for annotators.

**Diverse Representation:** Enhances inclusivity, reduces bias.
Varied Demographics: Text from diverse groups, languages, dialects, cultural norms.
Balanced Data Sources: News, social media, literature, technical documents.
Regional/Linguistic Variety: Improves global applicability, translation.

