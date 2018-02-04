# pttr means Potter!

Sources from this article can be found [here](https://github.com/mka-rainmaker/nlp-ptr).

## A machine learning model to understand fancy abbreviations 

How to recover phrases from abbreviations/typos, e.g. turn wtrbtl into water bottle, and bsktball into basketball. And there is an additional complication: lack of a comprehensive list of words. 
That means, we need an algorithm able to invent new likely words.

We were intrigued and started researching, which algorithms and math lie behind modern spell-checkers. It turned out that a good spell-checker can be made with an n-gram language model, a model of word distortions, and a greedy beam search algorithm. The whole construction is called a <a href="http://web.stanford.edu/~jurafsky/slp3/5.pdf">noisy channel </a> model.

## Math behind a spell checker

In the noisy channel model, each abbreviation is the result of a random distortion of the original phrase.

To recover the original phrase, we need to answer two questions: which original phrases are likely, and which distortions are likely?

By the Bayes theorem,
<img src="https://render.githubusercontent.com/render/math?math=p%28phrase%7Cabbreviation%29%20%5Csim%20p%28phrase%29%20p%28abbreviation%7Cphrase%29%20%3D%20%5C%5C%20%3D%20p%28phrase%29%20%5Csum%20p%28distortion%7Cphrase%29&mode=display" />
The $\sim$ symbol means "proportional", because LHS is a probability distribution, but RHS is generally not.

Both original phrase likelihood and distortion likelihood can be estimated with statistical models. We will use the simplest models - character [n-grams](https://en.wikipedia.org/wiki/N-gram). 
We could use more difficult models (e.g recurrent neural networks), but it doesn't change the principle.

With such models, we can reconstruct probable original phrases letter by letter, using a greedy directed search algorithm.


## N-gram language model

N-gram model looks at the previous n-1 letters and estimates the probability of the next (n'th) letter conditional on them. For example, the probability of letter "g" appearing after "bowlin" sequence would be calculated by 4-gram model as $p(g|bowlin)=p(g|lin)$, because the model ignores all the characters before these 4, for the sake of simplicity. Conditional probabilities, such as this, are determined ("learned") on a training corpus of texts. In my example,

<img src="https://render.githubusercontent.com/render/math?math=p%28g%7Clin%29%3D%5Cfrac%7B%5C%23%28ling%29%7D%7B%5C%23%28lin%5Cbullet%29%7D%3D%5Cfrac%7B%5C%23%28ling%29%7D%7B%5C%23%28lina%29%2B%5C%23%28linb%29%2B%5C%23%28linc%29%2B...%7D&mode=display" />

Here #(ling) is the number of occurrences of "ling" in the training text. $\#(lin\bullet)$ is the number of all 4-grams in the text, starting with "lin".

In order to estimate correctly even the rare n-grams, we apply two tricks. First, for each counter, we add a positive number $\delta$. It guarantees that we will not divide by zero. Second, we use not only n-grams (which can occur rarely in the text), but also n-1 grams (more frequent), and so on, down to 1-grams (unconditional probabilities of letters). But we discount lesser-order counters with an $\alpha$ multiplier. 
Thus, in fact we calculate <img src="https://render.githubusercontent.com/render/math?math=p%28g%7Clin%29&mode=inline" /> as
<img src="https://render.githubusercontent.com/render/math?math=p%28g%7Clin%29%3D%5Cfrac%7B%28%5C%23%28ling%29%2B1%29%20%2B%20%5Calpha%20%28%5C%23%28ing%29%2B1%29%20%2B%20%5Calpha%5E2%20%28%5C%23%28ng%29%2B1%29%20%2B%20%5Calpha%5E3%20%28%5C%23%28g%29%2B1%29%7D%7B%28%5C%23%28lin%5Cbullet%29%2B1%29%20%2B%20%5Calpha%20%28%5C%23%28in%5Cbullet%29%2B1%29%20%2B%20%5Calpha%5E2%20%28%5C%23%28n%5Cbullet%29%2B1%29%20%2B%20%5Calpha%5E3%20%28%5C%23%28%5Cbullet%29%2B1%29%7D&mode=display" />

## Greedy search for the most probable phrase

Having models of language and distortions, theoretically, we can estimate the likelihood of an original phrase. But for this, we need to loop over all the possible (original phrase, distortion) pairs. There are just too many of them: e.g. with 27 character alphabet, there are  $27^{10}$ possible 10-letter phrases. We need a smarter algorithm to avoid this near-infinite looping.

We will exploit the fact that the models are single-character-based, and will construct the phrase letter by letter. we will make a heap of incomplete candidate phrases, and evaluate the likelihood of each. The best candidate will be extended with multiple possible one-letter continuations and added to the heap. To cut the number of options, we will save only the "good enough" candidates. The complete candidates will be set aside, to be returned as a solution in the end. The procedure will be repeated unless either the heap or the maximum number of iterations run out.

The quality of candidates will be evaluated as log-probability of the abbreviation, given that the original phrase begins with the candidate and ends (because the candidate is incomplete) as the abbreviation itself. To manage the search, we introduced two parameters: "optimism" and "freedom". "Optimism" evaluates, how the likelihood will improve when the candidate completes. It makes sense to set "optimism" between 0 and 1; the closer it is to 1, the faster the algorithm will try to add new characters. "Freedom" is the allowable loss of quality in comparison to the current best candidate. The higher the "freedom", the more options would be included, and the slower the algorithm would be. If the "freedom" is too low, the heap may deplete before any reasonable phrase is found.

## Enough theory! Let's test our solution

To really test the algorithm, we need a good language model. We were wondering, how well a model could decipher the abbreviations if it had been trained on a deliberately limited corpus - one book on an unusual topic. The first such book that got into our hands was "Harry Potter and the Deathly Hallows". 

Well, let's see how well the magic language can help to decipher the modern sports terms.

Initialize our classes and "read" the book

```javascript
    var text = File.ReadAllText(@".\Data\Harry Potter and the Deathly Hallows.txt");
    LanguageModel langModel = new LanguageModel();

    Regex rgx = new Regex("[^a-zA-Z ]");
    text = rgx.Replace(text, "");

    langModel.Initialize(text.ToLowerInvariant());

    MissingLetterModel missingModel = new MissingLetterModel();
    var allLetters = string.Join("", langModel.Vocabulary.ToArray());
    var substitutions = new String('-', allLetters.Length);

    var data = new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>(allLetters, substitutions),
                new KeyValuePair<string, string>(allLetters, allLetters),
                new KeyValuePair<string, string>("aeiouy", "------"),
            };
    missingModel.Initialize(data);
```

And the code to generate candidates
```javascript
    var candidates = Functions.NoisyChannel(toRecover, langModel, missingModel);
```

Let's start our tests with **wtrbtl**
Higher score means less probability

| Suggestion| Score|
|--|--|
|water bottle | 27.7505636663187 |
|water but all|29.0722376312201|

What about **prb**?

|Suggestion|Score  |
|--|--|
| probably | 14.5220429153932 |
|problem|15.4457447531573|

And the very final test - **pttr** - returns only one single suggestion

|Suggestion|Score  |
|--|--|
| potter| 11.7895748701286 |

## Conclusions

Natural language processing is a complex mixture of science, technology, and magic. Even linguistic scientists cannot fully understand the laws of human speech. The times when machines indeed understand texts are not to come soon.

Natural language processing is also fun. Armed with a couple of statistical models, you can both recognize and generate non-obvious abbreviations.
