# Evaluation Results

## Test setup

- Dataset: 100 synthetic product-description examples
- Holdout set: 15 examples
- Compared examples: 10
- Base model: DistilGPT-2
- Fine-tuning: LoRA adapter

## Results

| Metric | Result |
|---|---:|
| Base model mean specification coverage | 6.7% |
| Fine-tuned raw model mean specification coverage | 10.0% |
| Final output mean specification coverage | 100.0% |
| Comparisons where fallback was used | 10/10 |

Specification coverage measures the fraction of supplied specification values appearing verbatim in the generated text.

## Interpretation

The fine-tuned model showed only a small improvement over the base model on this lexical metric. Its raw generations frequently omitted product names or specification values, and sometimes introduced unsupported details.

To avoid presenting unreliable generations as finished product descriptions, the application uses a specification-only fallback when the generated text fails its grounding checks.

The 100% final-output coverage is achieved by copying the supplied product facts into the fallback output. It is not a measure of the language model's quality or factual accuracy.

## Example

**Input product:** Classic Wireless Headphones 02

- Sound: Bluetooth 5.2
- Battery: Up to 24 hours
- Design: Lightweight on-ear

**Raw fine-tuned output:**

> Classic Wireless Headphones 02 offers 24 hour battery and up for24hour batteries with 24hours charging as well offering 48 Hours daily alongside 25 minutes everyday while showcasing LED activity in its colourful colour scheme.

**Application fallback output:**

> Classic Wireless Headphones 02 (Electronics) - Sound: Bluetooth 5.2; Battery: Up to 24 hours; Design: Lightweight on-ear.

The fallback preserves the supplied specifications instead of displaying the unreliable raw generation as the final result.

## Limitations and next steps

- The dataset is synthetic and small.
- Only 10 examples were used for the reported comparison.
- Specification coverage is a lexical metric, not a measure of fluency, relevance, or factual accuracy.
- Human review scores have not yet been collected.
- Further work should improve the training data, generation quality, and human evaluation.
