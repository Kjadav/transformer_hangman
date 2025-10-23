
guess_model_transformer(
    model,
    vocab,
    masked_input,
    guessed_letters
)

To explain the guess functions that i have used, i have gained inspiration from the following sources, combining white-papers knowledge, sourcing different information from stackoverflow and repositories. 

In terms of setting up the guessing_model_transformer, I knew there are masked inputs for which is _ _ _ etc. 
I had gained inspiration from this repo for starting off :
https://github.com/keras-team/keras-io/blob/master/examples/nlp/masked_language_modeling.py

Despite being a different language, i am able to utilise the concepts that being used here. I have set up the first part of the code, where I call in the model, and the vocab.encode which is to convert the input and assign integer value to the positions in the index, and then converting into dimensional that is usable. I apply the attention mask that is able to divert the attention of the model onto values that are not masked, to be able to use the model by having unmasked letters to predict the letter(assign probabilities).

We then iterate in range of 1.5-2 to assign even numbers for the positions in the masked positions and then creating assigning weighting in the dimensions that allows us to favour the desired probability in the set.

Next, we move onto the part we are sorting, assigning the weighting for the usage of cumulative probability that is being set for the letters that we can guess.
So in index_scores, we are essentially assigning the weighting to the list of the dimension. Once we have the sorted list of position and scores within that dimension. It would look like this [0.2, 0.4, 0.1, 0.3] and [0,2,1,3] for example. 

Now, we will move on the cumulative probability, where I have put a cut_off of 0.9. This is concept that I had came across when trying to increase my model. I had learnt of near-greedy encoding,top-k within reinforcement-learning, but was too difficult and longer for me to create. I wanted to go for an ML/Maths based approach, hence in the search, I came across top-p in : https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832. In search of trying to create my own top_p, credits to http://zeta.apac.ai/en/latest/zeta/utils/top_p/: 
I had just used the concept here. Having a cutoff of 0.9(via trial&error) -> where if for index positions, where until c.p(cumulative probability) is greater than 0.9, we will select the letters to guess until i, as they are sampled for the 90% of the sampling. 

We select those indices, and then convert the index to character, to guess those letters. 
We do have the measure in place to check if the character is in guess_letters or not, so we do not double guess the same letters.
I was contemplating the use of a reward system just how it is in reinforcment-learning, however it became really over-engineered and made the model worse.

Lastly, we are returning the top-letter choices at random, as through trial and error, it was repeating the guesses. 

guess()
The guess() function is standard, as it just cleans the input and we call the function. However, I have added my own variation of favouring letters when the letters at the end have not been guessed. They most likely are suffixes, such "ily", "ent", "ex" etc. Hence i had created a list of those ending letters to allow for more letters with high probability to be chosen once filling in the end. So basically going from back to front.

References:
https://openreview.net/forum?id=xi6lie0SUr
https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf
https://github.com/Aditya-dom/trexquant_Hangman
https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf