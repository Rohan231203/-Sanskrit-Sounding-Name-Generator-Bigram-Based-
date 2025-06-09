# -Sanskrit-Sounding-Name-Generator-Bigram-Based-
This project implements a character-level bigram model in PyTorch to generate names that mimic the structure and phonetics of Sanskrit-like words. It is intended as a simple, educational exercise in statistical language modeling and basic gradient-based learning.


🧠 What This Project Does
Reads a corpus of Sanskrit-inspired names from words.txt

Cleans and normalizes the names to lowercase a-z

Trains a bigram model:

First using count-based statistics (frequency matrix + smoothing)

Then using gradient descent to learn weights for character transitions

Generates new names by sampling characters one-by-one from learned distributions

Evaluates the model using Negative Log Likelihood (NLL)

📦 Project Structure
File	Description
words.txt	Corpus of Sanskrit-like names
main.py	All model training, visualization, and generation logic
output.txt	Stores generated names for inspection

✨ Sample Outputs
Generated Names (Bigram Count Model):

nginx
Copy
Edit
sugita
bhayaka
anikara
ratyana
vimodha
maniti
jayanas
vahara
dakuta
kumina
Generated Names (Learned Weights via Gradient Descent):

nginx
Copy
Edit
nasharaya
vamiya
pratakana
anima
savitasa
radhaya
kumana
vahiti
kanishta
jayavira
While not always perfect, these names approximate Sanskrit-sounding syllabic patterns with some degree of realism.

🔍 Modeling Approach
1. Data Preprocessing
All words are converted to lowercase.

Non-alphabet characters are stripped using regex (re.sub).

Start and end tokens are denoted by ".".

2. Bigram Frequency Matrix (Count-Based)
A 27x27 matrix N is built where each cell (i, j) counts transitions from character i to j (26 letters + '.' token).

Add-1 smoothing is applied to avoid zero probabilities.

Transition matrix P is normalized row-wise to obtain conditional probabilities.

3. Evaluation
Average Negative Log-Likelihood (NLL) is computed to measure model fit on training data:

python
Copy
Edit
log_likelihood = sum(log(P[c1, c2])) over all character pairs
4. Learnable Bigram Model
A weight matrix W is initialized and trained with gradient descent.

Uses softmax over logits to predict the next character.

Trained using PyTorch and F.cross_entropy.

📉 Training Results
bash
Copy
Edit
Epoch 1, Loss: 3.792
Epoch 20, Loss: 2.837
Epoch 50, Loss: 2.481
Epoch 100, Loss: 2.386
Loss steadily decreases as the weight matrix W learns to model better transitions based on input character distributions.

⚠️ Limitations
Bigram Only: Considers only one character of context. Can't capture multi-character patterns or syllabic constraints.

Gibberish Risk: Many names may still sound awkward or unrealistic.

No Linguistic or Semantic Awareness: Doesn't understand Sanskrit etymology or phonotactics.

Overfitting Risk: With a small corpus, frequent bigrams dominate generation.

📌 Conclusion
This project shows that even simple statistical models like bigrams can produce semi-coherent name-like structures, especially when trained with both count-based methods and gradient descent.

However, quality and realism are limited—higher-order models or LSTM/Transformer-based approaches would be better for more authentic name generation.
